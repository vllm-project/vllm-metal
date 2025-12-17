# SPDX-License-Identifier: Apache-2.0
"""Integration test for vLLM server chat completions."""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

# Mark all tests in this module as requiring Metal and being slow
pytestmark = [pytest.mark.metal, pytest.mark.slow, pytest.mark.server]

# Small instruct model for testing
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def find_free_port():
    """Find a free port to use for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SERVER_HOST, 0))  # Bind to localhost only, not all interfaces
        s.listen(1)
        return s.getsockname()[1]


def wait_for_server(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for server to be ready by checking the health endpoint."""
    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)

    return False


def is_intelligible(text: str) -> bool:
    """Check if the text is intelligible (contains actual words).

    This function checks that the output contains recognizable words
    rather than gibberish or random characters.
    """
    if not text or len(text.strip()) == 0:
        return False

    # Check for common English words
    common_words = [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "and",
        "in",
        "that",
        "it",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "this",
        "from",
        "or",
        "have",
        "has",
        "had",
        "not",
        "but",
        "what",
        "all",
        "can",
        "will",
        "would",
        "could",
        "you",
        "your",
        "we",
        "they",
        "he",
        "she",
        "my",
        "me",
        "do",
        "if",
        "so",
        "just",
        "about",
        "which",
        "there",
        "when",
        "how",
        "I",
        "hello",
        "hi",
        "yes",
        "no",
        "please",
        "thank",
        "help",
        "like",
        "one",
        "two",
        "first",
        "here",
        "great",
        "good",
        "new",
    ]

    # Split text into words and match using word boundaries
    text_words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    common_words_set = {word.lower() for word in common_words}
    words_found = len(text_words & common_words_set)

    # Check that it has reasonable word structure
    # (contains letters, has word-like patterns)
    has_letter_sequences = bool(re.search(r"[a-zA-Z]{2,}", text))

    # At least 2 common words or has letter sequences and some words
    return words_found >= 2 or (has_letter_sequences and words_found >= 1)


class TestServerChatCompletions:
    """Test vLLM server chat completions endpoint."""

    @pytest.fixture(scope="class")
    def server_port(self):
        """Get a free port for the server."""
        return find_free_port()

    @pytest.fixture(scope="class")
    def vllm_server(self, server_port):
        """Start and manage the vLLM server."""
        # Build the server command
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL_NAME,
            "--host",
            SERVER_HOST,
            "--port",
            str(server_port),
            "--dtype",
            "float16",
            "--max-model-len",
            "512",
            "--trust-remote-code",
        ]

        env = os.environ.copy()
        # Enable eager mode for Metal backend - this disables graph compilation
        # which can be slow and may not be fully supported on MPS yet.
        # Eager mode ensures immediate execution of operations.
        env["VLLM_METAL_EAGER_MODE"] = "1"

        print(f"\nStarting vLLM server with command: {' '.join(cmd)}")

        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Wait for server to be ready
        if not wait_for_server(SERVER_HOST, server_port, timeout=30):
            # Get output for debugging
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
                print(f"Server output: {output.decode()}")
            except subprocess.TimeoutExpired:
                print("Timed out waiting for server output")
            except UnicodeDecodeError as e:
                print(f"Failed to decode server output: {e}")
            pytest.fail("vLLM server did not start within timeout")

        print(f"vLLM server is ready on port {server_port}")

        yield {"host": SERVER_HOST, "port": server_port, "process": process}

        # Cleanup: terminate server
        print("\nShutting down vLLM server...")
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Server did not terminate in time, force killing...")
            process.kill()
        except ProcessLookupError:
            print("Server process already terminated")
        except OSError as e:
            print(f"OS error during cleanup: {e}")
            process.kill()

    def test_chat_completions_basic(self, vllm_server):
        """Test basic chat completions request returns intelligible response."""
        url = f"http://{vllm_server['host']}:{vllm_server['port']}/v1/chat/completions"

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, how are you today?"}],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            assert response.status == 200
            result = json.loads(response.read().decode("utf-8"))

        # Verify response structure
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]

        content = result["choices"][0]["message"]["content"]
        print(f"\nResponse content: {content}")

        # Check that the output is intelligible
        assert is_intelligible(content), f"Response is not intelligible: {content}"

    def test_chat_completions_question(self, vllm_server):
        """Test that the model can answer a simple question."""
        url = f"http://{vllm_server['host']}:{vllm_server['port']}/v1/chat/completions"

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2 plus 2?"}],
            "max_tokens": 30,
            "temperature": 0.1,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            assert response.status == 200
            result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        print(f"\nQuestion response: {content}")

        # The response should be intelligible
        assert is_intelligible(content), f"Response is not intelligible: {content}"

    def test_chat_completions_multi_turn(self, vllm_server):
        """Test multi-turn conversation."""
        url = f"http://{vllm_server['host']}:{vllm_server['port']}/v1/chat/completions"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 30,
            "temperature": 0.1,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            assert response.status == 200
            result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        print(f"\nMulti-turn response: {content}")

        # The response should be intelligible
        assert is_intelligible(content), f"Response is not intelligible: {content}"
