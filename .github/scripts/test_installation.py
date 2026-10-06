# SPDX-License-Identifier: Apache-2.0
"""Release-install smoke test, independent of the source tree's ML dependencies.

Run with VLLM_METAL_TEST_FRESH_INSTALL=1 on an Apple Silicon Mac with uv on PATH.
Each channel downloads packages and model weights into its own temporary directory.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stop_process_group(process):
    """Reap the server and its workers, including on a test assertion or timeout."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


class InstallerArguments(unittest.TestCase):
    def test_missing_venv_path_fails_before_installing(self):
        for args in (["--venv"], ["--venv", ""], ["--venv", "--stable"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(ROOT / "install.sh"), *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--venv requires a directory", result.stderr)

    def test_help_accepts_channel_and_venv_in_either_order(self):
        for args in (
            ["--stable", "--venv", "a path with spaces"],
            ["--venv", "a path with spaces", "--dev"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(ROOT / "install.sh"), *args, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                self.assertIn("--venv PATH", result.stdout)


@unittest.skipUnless(
    os.environ.get("VLLM_METAL_TEST_FRESH_INSTALL") == "1",
    "set VLLM_METAL_TEST_FRESH_INSTALL=1 to download and test release installs",
)
class FreshInstallation(unittest.TestCase):
    def test_default_development_channel(self):
        self.check_install([])

    def test_stable_channel(self):
        self.check_install(["--stable"])

    def check_install(self, channel_args):
        self.assertEqual((platform.system(), platform.machine()), ("Darwin", "arm64"))
        uv = shutil.which("uv")
        self.assertIsNotNone(
            uv, "Install uv and put it on PATH before running this test"
        )
        guide = (ROOT / "docs/installation.md").read_text()
        section = guide.split("## Run your first request\n", 1)[1]
        blocks = re.findall(r"```bash\n(.*?)```", section, re.DOTALL)
        self.assertEqual(len(blocks), 2, "Expected one server and one curl example")
        server_args, curl_args = [
            shlex.split(block.replace("\\\n", "")) for block in blocks
        ]
        self.assertEqual(server_args[0], "vllm")
        self.assertEqual(curl_args[0], "curl")
        request = json.loads(curl_args[curl_args.index("-d") + 1])

        # Keep IPC socket paths below macOS sockaddr_un.sun_path limits.
        with tempfile.TemporaryDirectory(
            prefix="metal-install-", dir="/tmp"
        ) as directory:
            work = Path(directory)
            venv = work / "fresh venv"
            # Keep HOME unchanged; isolate package, Python, and model caches.
            # Do not inherit an activated venv, Python path, HF token, or proxies.
            env = {
                "HOME": os.environ["HOME"],
                "PATH": f"{Path(uv).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": str(work),
                "UV_CACHE_DIR": str(work / "uv-cache"),
                "UV_PYTHON_INSTALL_DIR": str(work / "python"),
                "UV_PYTHON_PREFERENCE": "only-managed",
                "UV_NO_CONFIG": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HOME": str(work / "huggingface"),
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "DO_NOT_TRACK": "1",
            }
            self.assertFalse(venv.exists())
            install_log = work / "install.log"
            server_log = work / "server.log"
            try:
                with install_log.open("w") as log:
                    # stdin exercises the release-wheel branch used by curl | bash;
                    # ./install.sh in the checkout instead builds editable sources.
                    installer = subprocess.Popen(
                        ["bash", "-s", "--", *channel_args, "--venv", str(venv)],
                        stdin=subprocess.PIPE,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=work,
                        env=env,
                        text=True,
                        start_new_session=True,
                    )
                    try:
                        installer.communicate(
                            (ROOT / "install.sh").read_text(), timeout=1800
                        )
                        self.assertEqual(installer.returncode, 0, "installer failed")
                    finally:
                        stop_process_group(installer)
                self.assertTrue((venv / "bin/vllm").is_file())
                activated = subprocess.check_output(
                    [
                        "bash",
                        "-c",
                        'source "$1/bin/activate"; command -v vllm',
                        "activation-check",
                        str(venv),
                    ],
                    cwd=work,
                    env=env,
                    text=True,
                    timeout=10,
                ).strip()
                self.assertEqual(
                    Path(activated).resolve(), (venv / "bin/vllm").resolve()
                )
                python = venv / "bin/python"
                metadata = subprocess.check_output(
                    [
                        str(python),
                        "-c",
                        "from importlib.metadata import version; "
                        "print('vllm=' + version('vllm')); "
                        "print('vllm-metal=' + version('vllm-metal'))",
                    ],
                    cwd=work,
                    env=env,
                    text=True,
                    timeout=30,
                )
                print(metadata, flush=True)
                # Avoid conflicting with a user's running server. All other
                # serving flags and the request payload come from the guide.
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                old_port = server_args[server_args.index("--port") + 1]
                server_args[0] = str(venv / "bin/vllm")
                server_args[server_args.index("--port") + 1] = str(port)
                curl_args = [
                    arg.replace(f"127.0.0.1:{old_port}/", f"127.0.0.1:{port}/")
                    for arg in curl_args
                ]
                with server_log.open("w") as log:
                    server = subprocess.Popen(
                        server_args,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=work,
                        env=env,
                        start_new_session=True,
                    )
                    try:
                        deadline = time.monotonic() + 900
                        while True:
                            self.assertIsNone(
                                server.poll(), "server exited before readiness"
                            )
                            try:
                                with urllib.request.urlopen(
                                    f"http://127.0.0.1:{port}/health", timeout=2
                                ) as response:
                                    if response.status == 200:
                                        break
                            except (urllib.error.URLError, TimeoutError):
                                pass
                            self.assertLess(
                                time.monotonic(), deadline, "server readiness timed out"
                            )
                            time.sleep(1)
                        result = subprocess.run(
                            curl_args,
                            capture_output=True,
                            text=True,
                            check=True,
                            cwd=work,
                            env=env,
                            timeout=120,
                        )
                        reply = json.loads(result.stdout)
                        self.assertEqual(reply["model"], request["model"])
                        self.assertTrue(
                            reply["choices"][0]["message"]["content"].strip()
                        )
                        self.assertIn(
                            reply["choices"][0]["finish_reason"], ("stop", "length")
                        )
                        self.assertGreater(reply["usage"]["completion_tokens"], 0)
                        logs = server_log.read_text()
                        self.assertIn("Platform plugin metal is activated", logs)
                        self.assertIn("Device(gpu, 0)", logs)
                        print(json.dumps(reply), flush=True)
                    finally:
                        stop_process_group(server)
            except Exception:
                # Diagnostics survive TemporaryDirectory cleanup in the test log.
                for path in (install_log, server_log):
                    if path.exists():
                        print(
                            f"\n{path.name} (last 12000 characters):\n{path.read_text()[-12000:]}",
                            flush=True,
                        )
                raise


if __name__ == "__main__":
    unittest.main()
