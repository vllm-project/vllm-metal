#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal compatibility checker for text models on vllm-metal.

Checks three levels:
- Loadable: offline generate succeeds
- Usable: OpenAI-compatible chat completions succeeds
- Backend-Compatible: run the above checks on MLX and paged-attention paths

Execution policy:
- Prompts are executed one-by-one instead of as a batch.
- The safe default is a single model on the MLX path only.
- In multi-model or ``all`` mode, execution is serialized by stage/backend:
  native_mlx for all models, then mlx for all models, then paged_attention.
- Each stage performs ``gc.collect()`` and sleeps for ``--cooldown-seconds``
  to reduce Metal / unified-memory pressure on Apple Silicon.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Answer with exactly one word: blue.",
    "List three fruits, one per line.",
    'Return a JSON object with keys "city" and "country" for Paris.',
    "Explain gravity in one sentence for a 10-year-old.",
]

PROMPT_LABELS = {
    "What is the capital of France?": "capital",
    "Answer with exactly one word: blue.": "blue",
    "List three fruits, one per line.": "fruits",
    'Return a JSON object with keys "city" and "country" for Paris.': "json",
    "Explain gravity in one sentence for a 10-year-old.": "gravity",
}

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "openai/gpt-oss-20b",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "mlx-community/GLM-4.7-Flash-4bit",
]

DEFAULT_SAFE_MODEL = "all"


@dataclass
class StepResult:
    name: str
    ok: bool
    details: str
    elapsed_sec: float
    extra: dict[str, Any] | None = None


def _build_env(use_paged_attention: bool, paged_memory_fraction: str) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_METAL_USE_PAGED_ATTENTION"] = "1" if use_paged_attention else "0"
    env["VLLM_METAL_MEMORY_FRACTION"] = (
        paged_memory_fraction if use_paged_attention else "auto"
    )
    # Xet downloads have been intermittently returning HTTP 416 for some
    # public repos; plain hub downloads are slower but more reliable here.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def _build_native_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


def _nonempty_text(text: str) -> bool:
    return bool(text and text.strip())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = _strip_code_fences(text)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _count_sentences(text: str) -> int:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", _normalize_text(text))
        if part.strip()
    ]
    return len(parts)


def _validate_case(prompt: str, text: str) -> str | None:
    normalized = _normalize_text(text)
    lowered = normalized.lower()

    if prompt == "What is the capital of France?":
        return None if "paris" in lowered else "expected answer to mention Paris"

    if prompt == "Answer with exactly one word: blue.":
        cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", normalized).lower()
        if cleaned != "blue":
            return "expected exactly one word: blue"
        extra_words = re.findall(r"[A-Za-z]+", normalized)
        return None if extra_words == ["blue"] else "expected exactly one word: blue"

    if prompt == "List three fruits, one per line.":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 3:
            return f"expected exactly 3 non-empty lines, got {len(lines)}"
        return None

    if prompt == 'Return a JSON object with keys "city" and "country" for Paris.':
        obj = _extract_first_json_object(text)
        if obj is None:
            return "expected a parseable JSON object"
        if set(obj.keys()) != {"city", "country"}:
            return f'expected keys {{"city", "country"}}, got {set(obj.keys())}'
        city = str(obj.get("city", "")).strip().lower()
        country = str(obj.get("country", "")).strip().lower()
        if city != "paris" or country != "france":
            return 'expected {"city": "Paris", "country": "France"}'
        return None

    if prompt == "Explain gravity in one sentence for a 10-year-old.":
        if _count_sentences(text) != 1:
            return "expected exactly one sentence"
        if not any(token in lowered for token in ("gravity", "pull", "pulls")):
            return "expected answer to mention gravity or pulling"
        return None

    return None


def _truncate_detail(text: str, limit: int = 72) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _render_ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        cells = [cell.ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _status_text(value: bool | None) -> str:
    if value is None:
        return "no tested"
    return "yes" if value else "no"


def _status_with_source_text(value: bool | None, source: str | None = None) -> str:
    base = _status_text(value)
    if source and source != "no tested" and base != "no tested":
        return f"{source}: {base}"
    return base


def _content_ratio_text(total: int, passed: int, source: str | None = None) -> str:
    if total <= 0:
        return "no tested"
    prefix = f"{source}: " if source and source != "no tested" else ""
    return f"{prefix}{passed}/{total} passed"


def _step_prompt_ratio(step: dict[str, Any] | None) -> tuple[int, int] | None:
    if not step:
        return None
    extra = step.get("extra") or {}
    outputs = extra.get("outputs") or {}
    failures = extra.get("failures") or []
    total = len(outputs)
    if total <= 0:
        return None
    passed = total - len(failures)
    if passed < 0:
        passed = 0
    return total, passed


def _step_nonempty_ratio(step: dict[str, Any] | None) -> tuple[int, int] | None:
    if not step:
        return None
    extra = step.get("extra") or {}
    outputs = extra.get("outputs") or {}
    nonempty_failures = extra.get("nonempty_failures") or []
    total = len(outputs)
    if total <= 0:
        return None
    passed = total - len(nonempty_failures)
    if passed < 0:
        passed = 0
    return total, passed


def _normalize_chat_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("reasoning")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning"):
            text = value.get(key)
            if isinstance(text, str):
                return text
    return ""


def _extract_chat_message_text(response: dict[str, Any], model: str) -> str:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {}) or {}

    primary = _normalize_chat_field(message.get("content"))
    if _nonempty_text(primary):
        return primary

    if "gpt-oss" in model.lower():
        for candidate in (
            message.get("reasoning_content"),
            message.get("reasoning"),
            choice.get("reasoning_content"),
            choice.get("reasoning"),
        ):
            text = _normalize_chat_field(candidate)
            if _nonempty_text(text):
                return text

    return primary


def _find_step(
    model_report: dict[str, Any], backend_name: str, step_name: str
) -> dict[str, Any] | None:
    for backend_result in model_report.get("results", []):
        if backend_result.get("backend") != backend_name:
            continue
        for step in backend_result.get("steps", []):
            if step.get("name") == step_name:
                return step
    return None


def _preferred_quality_step(model_report: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    paged_step = _find_step(model_report, "paged_attention", "offline_generate")
    if paged_step is not None:
        return "paged_attention", paged_step

    mlx_step = _find_step(model_report, "mlx", "offline_generate")
    if mlx_step is not None:
        return "mlx", mlx_step

    native = model_report.get("native_mlx")
    if native is not None:
        return "native_mlx", native

    return "no tested", None


def _case_status_map(step: dict[str, Any] | None) -> dict[str, str]:
    statuses = {prompt: "no tested" for prompt in DEFAULT_PROMPTS}
    if not step:
        return statuses
    extra = step.get("extra") or {}
    outputs = extra.get("outputs") or {}
    failures = {
        item.get("prompt"): item.get("reason")
        for item in (extra.get("failures") or [])
        if item.get("prompt")
    }
    for prompt in DEFAULT_PROMPTS:
        if prompt not in outputs:
            continue
        statuses[prompt] = "no" if prompt in failures else "yes"
    return statuses


def _build_summary_table(report: dict[str, Any]) -> str:
    headers = [
        "Model",
        "mlx-lm/vlm runnable",
        "vllm-metal runnable",
        "vllm-metal paged runnable",
        "non-empty output",
        "content correctness",
    ]
    rows: list[list[str]] = []

    for model_report in report.get("reports", []):
        native = model_report.get("native_mlx")
        native_run_ok = None
        if native is not None:
            native_run_ok = bool((native.get("extra") or {}).get("run_ok"))

        mlx_run_checks: list[bool] = []
        paged_run_checks: list[bool] = []
        quality_source, quality_step = _preferred_quality_step(model_report)

        for backend_result in model_report.get("results", []):
            backend_name = backend_result.get("backend")
            step_run_values: list[bool] = []
            for step in backend_result.get("steps", []):
                extra = step.get("extra") or {}
                step_run_values.append(bool(extra.get("run_ok", False)))
            if backend_name == "mlx" and step_run_values:
                mlx_run_checks.extend(step_run_values)
            if backend_name == "paged_attention" and step_run_values:
                paged_run_checks.extend(step_run_values)

        vllm_mlx_run_ok = None if not mlx_run_checks else all(mlx_run_checks)
        paged_run_ok = None if not paged_run_checks else all(paged_run_checks)
        nonempty_total, nonempty_passed = _step_nonempty_ratio(quality_step) or (0, 0)
        content_total, content_passed = _step_prompt_ratio(quality_step) or (0, 0)

        rows.append(
            [
                str(model_report.get("model", "")),
                _status_text(native_run_ok),
                _status_text(vllm_mlx_run_ok),
                _status_text(paged_run_ok),
                _content_ratio_text(nonempty_total, nonempty_passed, quality_source),
                _content_ratio_text(content_total, content_passed, quality_source),
            ]
        )

    return _render_ascii_table(headers, rows)


def _write_report_snapshot(report: dict[str, Any], output_json: str | None) -> None:
    if not output_json:
        return
    text = json.dumps(report, ensure_ascii=False, indent=2)
    with open(output_json, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def run_native_mlx_generate(
    *,
    model: str,
    trust_remote_code: bool,
    max_tokens: int,
) -> StepResult:
    start = time.time()
    old_env = os.environ.copy()
    os.environ.update(_build_native_env())
    try:
        backend_used = "mlx_lm"
        model_obj = None
        tokenizer_or_processor = None
        load_errors: list[str] = []

        try:
            from mlx_lm import generate as mlx_lm_generate
            from mlx_lm import load as mlx_lm_load

            model_obj, tokenizer_or_processor = mlx_lm_load(
                model,
                tokenizer_config={"trust_remote_code": trust_remote_code},
            )

            def native_generate(prompt: str) -> str:
                text = mlx_lm_generate(
                    model_obj,
                    tokenizer_or_processor,
                    prompt,
                    verbose=False,
                    max_tokens=max_tokens,
                )
                return text or ""

        except Exception as exc:
            load_errors.append(f"mlx_lm: {type(exc).__name__}: {exc}")
            from mlx_vlm import generate as mlx_vlm_generate
            from mlx_vlm import load as mlx_vlm_load

            backend_used = "mlx_vlm"
            model_obj, tokenizer_or_processor = mlx_vlm_load(
                model,
                trust_remote_code=trust_remote_code,
            )

            def native_generate(prompt: str) -> str:
                result = mlx_vlm_generate(
                    model_obj,
                    tokenizer_or_processor,
                    prompt,
                    verbose=False,
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return result.text if getattr(result, "text", None) else ""

        rendered: dict[str, str] = {}
        for prompt in DEFAULT_PROMPTS:
            rendered[prompt] = native_generate(prompt)
        nonempty_failures = []
        validation_failures = []
        for prompt in DEFAULT_PROMPTS:
            text = rendered.get(prompt, "")
            if not _nonempty_text(text):
                nonempty_failures.append({"prompt": prompt, "reason": "empty output"})
                continue
            error = _validate_case(prompt, text)
            if error:
                validation_failures.append({"prompt": prompt, "reason": error})
        failures = nonempty_failures + validation_failures
        run_ok = len(rendered) == len(DEFAULT_PROMPTS)
        nonempty_ok = run_ok and not nonempty_failures
        content_ok = nonempty_ok and not validation_failures
        ok = run_ok
        details = (
            f"native {backend_used} generate satisfied all case-specific checks"
            if content_ok
            else f"native {backend_used} validation failures: {failures}"
        )
        return StepResult(
            name="native_mlx_generate",
            ok=ok,
            details=details,
            elapsed_sec=time.time() - start,
            extra={
                "native_backend": backend_used,
                "outputs": rendered,
                "failures": failures,
                "nonempty_failures": nonempty_failures,
                "validation_failures": validation_failures,
                "run_ok": run_ok,
                "nonempty_ok": nonempty_ok,
                "content_ok": content_ok,
                "load_errors": load_errors,
            },
        )
    except Exception as exc:
        return StepResult(
            name="native_mlx_generate",
            ok=False,
            details=f"{type(exc).__name__}: {exc}",
            elapsed_sec=time.time() - start,
            extra={"traceback": traceback.format_exc()},
        )
    finally:
        gc.collect()
        os.environ.clear()
        os.environ.update(old_env)


def run_offline_generate(
    *,
    model: str,
    trust_remote_code: bool,
    max_model_len: int,
    max_tokens: int,
    use_paged_attention: bool,
    paged_memory_fraction: str,
) -> StepResult:
    start = time.time()
    env = _build_env(use_paged_attention, paged_memory_fraction)
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            max_num_seqs=1,
        )
        rendered: dict[str, str] = {}
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        for prompt in DEFAULT_PROMPTS:
            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0] if outputs else None
            rendered[prompt] = (
                output.outputs[0].text if output and output.outputs else ""
            )
        nonempty_failures = []
        validation_failures = []
        for prompt in DEFAULT_PROMPTS:
            text = rendered.get(prompt, "")
            if not _nonempty_text(text):
                nonempty_failures.append({"prompt": prompt, "reason": "empty output"})
                continue
            error = _validate_case(prompt, text)
            if error:
                validation_failures.append({"prompt": prompt, "reason": error})
        failures = nonempty_failures + validation_failures
        run_ok = len(rendered) == len(DEFAULT_PROMPTS)
        nonempty_ok = run_ok and not nonempty_failures
        content_ok = nonempty_ok and not validation_failures
        ok = run_ok
        details = (
            "offline generate satisfied all case-specific checks"
            if content_ok
            else f"validation failures for prompts: {failures}"
        )
        return StepResult(
            name="offline_generate",
            ok=ok,
            details=details,
            elapsed_sec=time.time() - start,
            extra={
                "outputs": rendered,
                "failures": failures,
                "nonempty_failures": nonempty_failures,
                "validation_failures": validation_failures,
                "run_ok": run_ok,
                "nonempty_ok": nonempty_ok,
                "content_ok": content_ok,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised in real runs
        return StepResult(
            name="offline_generate",
            ok=False,
            details=f"{type(exc).__name__}: {exc}",
            elapsed_sec=time.time() - start,
            extra={"traceback": traceback.format_exc()},
        )
    finally:
        gc.collect()
        os.environ.clear()
        os.environ.update(old_env)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(base_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # pragma: no cover - real runtime path
            last_error = str(exc)
            time.sleep(1.0)
    raise TimeoutError(f"server healthcheck did not become ready: {last_error}")


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _drain_log_tail(proc: subprocess.Popen[str], max_chars: int = 4000) -> str:
    if proc.stdout is None:
        return ""
    try:
        text = proc.stdout.read()
    except Exception:
        return ""
    return text[-max_chars:]


def run_chat_api(
    *,
    model: str,
    trust_remote_code: bool,
    max_model_len: int,
    max_tokens: int,
    use_paged_attention: bool,
    paged_memory_fraction: str,
    startup_timeout: float,
) -> StepResult:
    start = time.time()
    env = _build_env(use_paged_attention, paged_memory_fraction)
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    vllm_bin = shutil.which("vllm") or str(Path(sys.executable).with_name("vllm"))
    cmd = [
        vllm_bin,
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_tail = ""
    try:
        _wait_for_health(base_url, startup_timeout)
        rendered: dict[str, str] = {}
        nonempty_failures = []
        validation_failures = []
        responses: dict[str, Any] = {}
        for prompt in DEFAULT_PROMPTS:
            response = _http_post_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=60.0,
            )
            responses[prompt] = response
            content = _extract_chat_message_text(response, model)
            rendered[prompt] = content or ""
            if not _nonempty_text(content):
                nonempty_failures.append({"prompt": prompt, "reason": "empty content"})
                continue
            validation_error = _validate_case(prompt, content)
            if validation_error:
                validation_failures.append({"prompt": prompt, "reason": validation_error})
        failures = nonempty_failures + validation_failures
        run_ok = len(rendered) == len(DEFAULT_PROMPTS)
        nonempty_ok = run_ok and not nonempty_failures
        content_ok = nonempty_ok and not validation_failures
        ok = run_ok
        details = (
            "chat completions satisfied all case-specific checks"
            if content_ok
            else f"chat completion validation failures: {failures}"
        )
        return StepResult(
            name="chat_api",
            ok=ok,
            details=details,
            elapsed_sec=time.time() - start,
            extra={
                "outputs": rendered,
                "failures": failures,
                "nonempty_failures": nonempty_failures,
                "validation_failures": validation_failures,
                "run_ok": run_ok,
                "nonempty_ok": nonempty_ok,
                "content_ok": content_ok,
                "responses": responses,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised in real runs
        _terminate_process(proc)
        log_tail = _drain_log_tail(proc)
        return StepResult(
            name="chat_api",
            ok=False,
            details=f"{type(exc).__name__}: {exc}",
            elapsed_sec=time.time() - start,
            extra={"traceback": traceback.format_exc(), "server_log_tail": log_tail},
        )
    finally:
        _terminate_process(proc)


def run_backend(
    *,
    model: str,
    trust_remote_code: bool,
    max_model_len: int,
    max_tokens: int,
    use_paged_attention: bool,
    paged_memory_fraction: str,
    run_api: bool,
    startup_timeout: float,
) -> dict[str, Any]:
    backend_name = "paged_attention" if use_paged_attention else "mlx"
    steps = [
        run_offline_generate(
            model=model,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            use_paged_attention=use_paged_attention,
            paged_memory_fraction=paged_memory_fraction,
        )
    ]
    if run_api:
        steps.append(
            run_chat_api(
                model=model,
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                max_tokens=max_tokens,
                use_paged_attention=use_paged_attention,
                paged_memory_fraction=paged_memory_fraction,
                startup_timeout=startup_timeout,
            )
        )
    return {
        "backend": backend_name,
        "ok": all(step.ok for step in steps),
        "steps": [asdict(step) for step in steps],
    }


def _cooldown(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def _stage_cleanup(seconds: float) -> None:
    gc.collect()
    _cooldown(seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Hugging Face model id. Repeat to run multiple models. "
            "Use 'all' to run the built-in model list. Omit this flag to run a single safe default model."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Model context length used for validation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Generation length used for validation",
    )
    parser.add_argument(
        "--paged-memory-fraction",
        default="0.3",
        help="VLLM_METAL_MEMORY_FRACTION used when paged attention is enabled",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "paged", "both"],
        default="both",
        help="Which backend path to validate. Default runs both mlx and paged.",
    )
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Also run chat/completions checks. Disabled by default for safety.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for vLLM server health",
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path to write the full JSON report",
    )
    parser.add_argument(
        "--skip-native-mlx",
        action="store_true",
        help="Skip direct mlx-lm/mlx-vlm validation",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=60.0,
        help="Seconds to sleep between validation stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_selectors = args.models or [DEFAULT_SAFE_MODEL]
    if "all" in model_selectors:
        model_ids = DEFAULT_MODELS
    else:
        model_ids = model_selectors
    backends = []
    if args.backend in ("mlx", "both"):
        backends.append(False)
    if args.backend in ("paged", "both"):
        backends.append(True)

    report = {
        "model_selector": "all" if "all" in model_selectors else model_ids,
        "models": model_ids,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "reports": [],
    }

    report_by_model: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        model_report = {
            "model": model_id,
            "native_mlx": None,
            "results": [],
        }
        report["reports"].append(model_report)
        report_by_model[model_id] = model_report

    def refresh_snapshot() -> None:
        for model_report in report["reports"]:
            native_ok = True
            if model_report["native_mlx"] is not None:
                native_ok = bool(model_report["native_mlx"]["ok"])
            model_report["ok"] = native_ok and all(
                result["ok"] for result in model_report["results"]
            )
        report["ok"] = all(
            partial_report.get("ok", False) for partial_report in report["reports"]
        )
        report["dimension_table"] = _build_summary_table(report)
        _write_report_snapshot(report, args.output_json)

    if not args.skip_native_mlx:
        for model_id in model_ids:
            report_by_model[model_id]["native_mlx"] = asdict(
                run_native_mlx_generate(
                    model=model_id,
                    trust_remote_code=args.trust_remote_code,
                    max_tokens=args.max_tokens,
                )
            )
            refresh_snapshot()
            _stage_cleanup(args.cooldown_seconds)
        _stage_cleanup(args.cooldown_seconds)

    for use_paged_attention in backends:
        for model_id in model_ids:
            report_by_model[model_id]["results"].append(
                run_backend(
                    model=model_id,
                    trust_remote_code=args.trust_remote_code,
                    max_model_len=args.max_model_len,
                    max_tokens=args.max_tokens,
                    use_paged_attention=use_paged_attention,
                    paged_memory_fraction=args.paged_memory_fraction,
                    run_api=args.enable_api,
                    startup_timeout=args.startup_timeout,
                )
            )
            refresh_snapshot()
            _stage_cleanup(args.cooldown_seconds)
        _stage_cleanup(args.cooldown_seconds)

    report["ok"] = all(model_report["ok"] for model_report in report["reports"])
    dimension_table = _build_summary_table(report)
    report["dimension_table"] = dimension_table
    print(dimension_table)
    print()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    _write_report_snapshot(report, args.output_json)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
