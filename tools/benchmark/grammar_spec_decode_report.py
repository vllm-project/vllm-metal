"""Benchmark grammar-forced speculative decoding against a plain baseline.

Built on the same bones as ``gemma4_mtp_report.py`` -- one mode per process,
JSON per run, Markdown rendered from a set of JSONs -- so the two reports stay
comparable line for line. The measurement helpers (timed passes, MLX/RSS memory,
Prometheus counters, machine metadata) are deliberately duplicated here rather
than imported from that module: this script depends only on
``gemma4_mtp_benchmark.py``, which is part of the repo, so it stands alone.

The one structural difference: this proposer needs no monkeypatch probes. It
keeps its own ``GrammarProposerStats`` counters, so the harness just reaches
into the live drafter and reads them -- which still requires
``VLLM_ENABLE_V1_MULTIPROCESSING=0``, since under multiprocessing the drafter
lives in another process.

Usage:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tools.benchmark.grammar_spec_decode_report \\
        run --model Qwen/Qwen3-0.6B --batch-size 1 --output-json run-grammar-bs1.json \\
        --grammar
    python -m tools.benchmark.grammar_spec_decode_report report run-*.json \\
        --output-md grammar-spec-decode-report.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmark.gemma4_mtp_benchmark import (  # noqa: E402
    _package_version,
    environment_metadata,
    select_prompts,
    summarize_outputs,
)

_PROM_DRAFTS = "vllm:spec_decode_num_drafts"
_PROM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
_PROM_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
_PROM_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"


def _gb(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value / 1e9:.2f} GB"


def _ms(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds * 1e3:.1f} ms"


def _peak_rss_bytes() -> int:
    """Peak resident set size. ``ru_maxrss`` is bytes on Darwin, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if platform.system() == "Darwin" else int(raw) * 1024


def _mlx_memory() -> dict[str, int]:
    import mlx.core as mx

    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
    }


def machine_metadata() -> dict[str, Any]:
    """Host facts that decide whether these numbers transfer anywhere else."""
    import mlx.core as mx

    def sysctl(name: str) -> str | None:
        try:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    info = mx.device_info()
    mem_size = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(mem_size) if mem_size and mem_size.isdigit() else None,
        "macos": platform.mac_ver()[0] or None,
        "mlx_device": info.get("device_name"),
        "max_recommended_working_set_size": info.get(
            "max_recommended_working_set_size"
        ),
        "max_buffer_length": info.get("max_buffer_length"),
    }


def _timed_pass(
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    *,
    repeats: int,
    capture_tokens: bool,
    sequential: bool = False,
) -> dict[str, Any]:
    """Run ``repeats`` identical generate() calls and summarise them.

    Only the last repeat's token ids are kept: under greedy sampling with a
    fixed seed every repeat emits the same ids, so keeping one is enough for
    the baseline-vs-grammar equality check and keeps the JSON small.

    ``sequential`` issues each prompt as its own ``generate()`` call instead of
    one batch. That exists for the ToolSpec arm and is the only setup in which
    retrieval can be measured honestly: its memory is populated by requests that
    have *finished*, so inside a single batch it is necessarily empty, and
    replaying one batch under ``--repeats`` would instead have it retrieve each
    prompt's own previous output -- memorisation, not generalisation. Timing is
    the sum over the sequence, so tokens/s stays comparable with the batched
    arms at the same batch size of 1.
    """
    elapsed: list[float] = []
    output_tokens = 0
    prompt_tokens = 0
    samples: list[dict[str, Any]] = []

    for _ in range(repeats):
        start = time.perf_counter()
        if sequential:
            outputs = []
            for prompt in prompts:
                outputs.extend(llm.generate([prompt], sampling_params, use_tqdm=False))
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed.append(time.perf_counter() - start)
        prompt_tokens, output_tokens, samples = summarize_outputs(
            outputs, include_text=False
        )

    mean_elapsed = statistics.fmean(elapsed)
    return {
        "repeats": repeats,
        "elapsed_s": elapsed,
        "mean_elapsed_s": mean_elapsed,
        "min_elapsed_s": min(elapsed),
        "stdev_elapsed_s": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_output_tokens_per_s": (
            output_tokens / mean_elapsed if mean_elapsed else 0.0
        ),
        "outputs": samples if capture_tokens else [],
    }


def _prom_metrics(llm: Any) -> dict[str, Any]:
    """Read vLLM's own spec-decode counters, or explain why we could not."""
    try:
        metrics = llm.get_metrics()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {"available": False, "reason": str(exc)}

    wanted = {
        _PROM_DRAFTS: "num_drafts",
        _PROM_DRAFT_TOKENS: "num_draft_tokens",
        _PROM_ACCEPTED: "num_accepted_tokens",
        _PROM_ACCEPTED_PER_POS: "num_accepted_tokens_per_pos",
    }
    out: dict[str, Any] = {"available": True}
    for metric in metrics:
        key = wanted.get(metric.name)
        if key is None:
            continue
        value = getattr(metric, "value", None)
        if value is None:
            value = getattr(metric, "values", None)
        if value is None:
            continue
        out[key] = list(value) if isinstance(value, (list, tuple)) else float(value)
    return out


SCHEMA = "vllm-metal.grammar-spec-decode-report/1"

PROPOSER_PATH = "vllm_metal.v1.grammar_proposer.GrammarProposer"
# The ToolSpec port: the same grammar drafting plus retrieval over past
# invocations, so the two arms isolate exactly what retrieval adds.
TOOLSPEC_PROPOSER_PATH = "vllm_metal.v1.toolspec_proposer.ToolSpecProposer"

# Arms compared against baseline in the report. n-gram is the control that
# matters most: it is already in-tree, so a new proposer has to beat it and
# not merely the baseline.
COMPARED_ARMS = ("ngram", "grammar", "toolspec")
# Arms backed by this repo's own proposers, whose per-step counters the
# coverage/cost tables read. n-gram keeps different counters.
PROPOSER_ARMS = ("grammar", "toolspec")

# A tool-call-shaped schema: a fixed skeleton the grammar determines, an enum
# the model picks from, and a free string it writes itself. The mix is the
# point -- a schema that is all skeleton would flatter the proposer, and one
# that is all free text would hide it.
DEFAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["get_weather", "get_time", "search_web", "send_email"],
        },
        "arguments": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "query": {"type": "string"},
            },
            "required": ["location", "unit", "query"],
        },
    },
    "required": ["name", "arguments"],
}

DEFAULT_PROMPTS = [
    "What is the weather in Paris? Use celsius. Emit a tool call as JSON.",
    "What is the weather in Tokyo? Use fahrenheit. Emit a tool call as JSON.",
    "What time is it in Sao Paulo right now? Emit a tool call as JSON.",
    "Search the web for the tallest building in Dubai. Emit a tool call as JSON.",
    "Email the quarterly report to the finance team. Emit a tool call as JSON.",
    "What is the weather in Reykjavik? Use celsius. Emit a tool call as JSON.",
    "Look up the current exchange rate for the yen. Emit a tool call as JSON.",
    "What time is it in Auckland? Emit a tool call as JSON.",
]


def _find_drafter(llm: Any) -> Any:
    """Reach the live proposer inside the in-process executor, or ``None``.

    Deliberately defensive: the path crosses four upstream objects whose names
    have changed before, and a missing drafter must degrade to "no stats", never
    to a crash in the middle of a benchmark run.
    """
    node: Any = llm
    for attr in (
        "llm_engine",
        "engine_core",
        "engine_core",
        "model_executor",
        "driver_worker",
        "model_runner",
        "_drafter",
    ):
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def _drafter_stats(llm: Any) -> dict[str, Any]:
    drafter = _find_drafter(llm)
    stats = getattr(drafter, "stats", None)
    if stats is None:
        return {"available": False, "reason": "no drafter with stats in this process"}
    record = {
        "available": True,
        "class": type(drafter).__name__,
        **{key: getattr(stats, key) for key in vars(stats)},
    }
    # ToolSpecProposer composes a GrammarProposer and a RetrievalStore, each
    # with its own counters. Nesting them keeps the top level comparable with
    # the `grammar` arm while still recording which half did the drafting.
    inner = getattr(drafter, "_grammar", None)
    if inner is not None and getattr(inner, "stats", None) is not None:
        record["grammar_half"] = {
            key: getattr(inner.stats, key) for key in vars(inner.stats)
        }
    store = getattr(drafter, "store", None)
    if store is not None and getattr(store, "stats", None) is not None:
        record["retrieval_store"] = {
            "size": len(store),
            "capacity": store.capacity,
            **{key: getattr(store.stats, key) for key in vars(store.stats)},
        }
    return record


def _llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "async_scheduling": False,
        # Offline LLM() defaults this to True, which makes get_metrics() assert.
        "disable_log_stats": False,
        # Identical prompts are replayed, so prefix caching would serve prefill
        # from cache after the first pass and report a warm TTFT as a cold one.
        "enable_prefix_caching": args.prefix_caching,
        # The backend must be pinned: `auto` refuses disable_any_whitespace, and
        # the proposer only drafts for xgrammar-routed requests anyway.
        "structured_outputs_config": {
            "backend": "xgrammar",
            "disable_any_whitespace": args.disable_any_whitespace,
        },
    }
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.arm == "grammar":
        kwargs["speculative_config"] = {
            "method": "custom_class",
            "model": PROPOSER_PATH,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    elif args.arm == "toolspec":
        # Same K and same grammar half as the `grammar` arm, so the difference
        # between the two is retrieval and nothing else.
        kwargs["speculative_config"] = {
            "method": "custom_class",
            "model": TOOLSPEC_PROPOSER_PATH,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    elif args.arm == "ngram":
        # The in-tree proposer, as the control for "why not just use ngram".
        # Same K, so the only difference is where the draft comes from.
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.num_speculative_tokens,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": args.num_speculative_tokens,
        }
    return kwargs


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    """Run one (mode, batch_size) configuration and return its record."""
    if args.arm != "baseline" and (
        os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0"
    ):
        raise RuntimeError(
            "The drafter's counters are read in-process, so this report requires "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0. Without it the drafter lives in "
            "another process and every coverage/acceptance number would be "
            "missing. Export it and re-run."
        )

    import mlx.core as mx
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = DEFAULT_SCHEMA
    if args.schema_file is not None:
        schema = json.loads(Path(args.schema_file).read_text())

    # In sequential mode the prompts are a *stream* of distinct requests rather
    # than one batch, so the count comes from --num-prompts and they must not be
    # cycled: repeating a prompt would let retrieval match a request against its
    # own earlier output.
    if args.sequential:
        raw_prompts = select_prompts(
            batch_size=args.num_prompts,
            prompts=args.prompt or list(DEFAULT_PROMPTS),
            prompt_file=args.prompt_file,
        )
        distinct = len(dict.fromkeys(raw_prompts))
        if distinct < len(raw_prompts):
            raise SystemExit(
                f"--sequential needs {len(raw_prompts)} distinct prompts but the "
                f"source has only {distinct}. Retrieval measured over repeated "
                "prompts is memorisation, not generalisation -- supply a larger "
                "--prompt-file or lower --num-prompts."
            )
    else:
        raw_prompts = select_prompts(
            batch_size=args.batch_size,
            prompts=args.prompt or list(DEFAULT_PROMPTS),
            prompt_file=args.prompt_file,
        )

    load_start = time.perf_counter()
    llm = LLM(**_llm_kwargs(args))
    load_s = time.perf_counter() - load_start
    after_load = _mlx_memory()

    prompts = raw_prompts
    if args.chat:
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if args.no_thinking else {}),
            )
            for prompt in raw_prompts
        ]

    schema_json = json.dumps(schema)

    def params(max_tokens: int) -> Any:
        # --no-schema is the unconstrained control: no grammar at all, which is
        # what ordinary chat traffic looks like. The grammar half must draft
        # nothing there and the step must stay on Metal's one-row decode fast
        # path, so this arm's job is to come out at 1.00x rather than fast.
        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=args.ignore_eos,
            structured_outputs=(
                None
                if args.no_schema
                else StructuredOutputsParams(json=schema_json)
            ),
        )

    if args.sequential and (args.warmup or args.repeats != 1):
        # Any second pass over the measured prompts leaves each one's own
        # output in the retrieval memory, and the next pass then "retrieves" it
        # verbatim. That is a memorisation number, not a generalisation one, so
        # it is refused rather than footnoted. Run the whole process repeatedly
        # and compare the JSONs for a noise estimate instead.
        raise SystemExit(
            "--sequential requires --warmup 0 --repeats 1: a second pass over "
            "the measured prompts pre-loads the retrieval memory with their "
            "own outputs. Re-run the process for repeat measurements."
        )
    for _ in range(args.warmup):
        llm.generate(prompts, params(args.max_tokens), use_tqdm=False)

    mx.reset_peak_memory()
    stats_before = _drafter_stats(llm)

    ttft = _timed_pass(
        llm,
        prompts,
        params(1),
        repeats=args.repeats,
        capture_tokens=False,
        sequential=args.sequential,
    )
    # Counters are sampled around the full pass only, so warmup and the TTFT
    # pass do not pollute the per-step drafter numbers.
    mid = _drafter_stats(llm)
    full = _timed_pass(
        llm,
        prompts,
        params(args.max_tokens),
        repeats=args.repeats,
        capture_tokens=True,
        sequential=args.sequential,
    )
    after = _drafter_stats(llm)

    steady: dict[str, Any] = {}
    if after.get("available") and mid.get("available"):
        for key, value in after.items():
            # `available` is a bool and bool subclasses int, so it has to be
            # excluded by name or the subtraction below turns it into 0.
            if key in ("available", "class", "reason"):
                continue
            if isinstance(value, (int, float)) and key in mid:
                steady[key] = value - mid[key]
            elif key == "skipped_backends":
                steady[key] = value
    steady["available"] = bool(after.get("available") and mid.get("available"))

    prom = _prom_metrics(llm)
    after_decode = _mlx_memory()
    peak_rss = _peak_rss_bytes()

    ttft_s = ttft["mean_elapsed_s"]
    # Batched mode decodes one step for the whole batch at a time, so a pass is
    # (max_tokens - 1) steps regardless of batch size. Sequential mode runs the
    # prompts end to end, so a pass is that many steps *per prompt*.
    decode_steps = (args.max_tokens - 1) * (len(prompts) if args.sequential else 1)
    tpot_s = (
        (full["mean_elapsed_s"] - ttft_s) / decode_steps if decode_steps > 0 else None
    )

    # _timed_pass keeps token ids but not text, so decode here rather than
    # spend another generate() pass just to look at the strings.
    tokenizer = llm.get_tokenizer()
    valid = 0
    for sample in full["outputs"]:
        try:
            json.loads(tokenizer.decode(sample["token_ids"], skip_special_tokens=True))
            valid += 1
        except Exception:  # noqa: BLE001 - a malformed sample is the finding
            pass

    del llm
    gc.collect()

    return {
        "schema": SCHEMA,
        "mode": args.arm,
        "label": args.label or args.arm,
        "config": {
            "model": args.model,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.arm != "baseline" else None
            ),
            "disable_any_whitespace": args.disable_any_whitespace,
            "ignore_eos": args.ignore_eos,
            "prefix_caching": args.prefix_caching,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "chat_template_applied": bool(args.chat),
            "sequential": bool(args.sequential),
            # In sequential mode this is the length of the prompt stream, which
            # is what a run is actually made of; batch_size stays 1.
            "num_prompts": len(prompts) if args.sequential else args.batch_size,
        },
        "json_schema": schema,
        "environment": {
            **environment_metadata(),
            "xgrammar": _package_version("xgrammar"),
        },
        "machine": machine_metadata(),
        "prompts": raw_prompts,
        "latency": {
            "model_load_s": load_s,
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "ttft_pass": ttft,
            "full_pass": full,
        },
        "memory": {
            "after_load": after_load,
            "after_decode": after_decode,
            "peak_rss_bytes": peak_rss,
        },
        "drafter": {
            "at_load": stats_before,
            "steady_state": steady,
            "cumulative": after,
            "prometheus": prom,
        },
        "schema_valid_outputs": valid,
    }


# -- reporting ---------------------------------------------------------------


def _pct(numerator: float, denominator: float) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def _coverage(record: dict[str, Any]) -> dict[str, Any]:
    """Coverage, acceptance and #MAT from the proposer's own counters.

    ``#MAT`` follows ToolSpec's definition: the mean number of tokens a decoding
    step commits, counting the always-committed bonus token. ``steps_eligible``
    is the per-request step count, so this is directly comparable with their
    Table 3 -- a value of 1.00 is plain autoregressive decoding.
    """
    steady = record.get("drafter", {}).get("steady_state", {})
    if not steady.get("available"):
        return {"available": False}
    eligible = steady.get("steps_eligible", 0)
    drafted = steady.get("steps_drafted", 0)
    offered = steady.get("drafts_offered", 0)
    accepted = steady.get("drafts_accepted", 0)
    return {
        "available": True,
        "steps_eligible": eligible,
        "steps_drafted": drafted,
        "coverage": drafted / eligible if eligible else 0.0,
        "offered": offered,
        "accepted": accepted,
        "acceptance": accepted / offered if offered else 0.0,
        "tokens_per_draft": offered / drafted if drafted else 0.0,
        "mat": (accepted + eligible) / eligible if eligible else 1.0,
        # The share of emitted tokens that came from a draft rather than from
        # their own forward. Step coverage understates the proposer's reach,
        # because the steps it does draft on carry several tokens each.
        "token_share": accepted / (accepted + eligible) if eligible else 0.0,
        "truncated": steady.get("truncated_drafts", 0),
        "altered": steady.get("altered_drafts", 0),
        "rejected": steady.get("rejected_drafts", 0),
    }


def render_markdown(records: Sequence[dict[str, Any]], notes: Sequence[str]) -> str:
    records = sorted(
        records,
        key=lambda r: (r["config"]["batch_size"], r["mode"] != "baseline"),
    )
    first = records[0]
    machine = first.get("machine", {})
    env = first.get("environment", {})

    lines: list[str] = []
    add = lines.append

    add("# Grammar-forced speculative decoding on Metal")
    add("")
    add(
        "Baseline vs. grammar-forced drafting "
        f"(`{PROPOSER_PATH}`) on structured-output requests."
    )
    add("")

    add("## Setup")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Model | `{first['config']['model']}` |")
    add(
        f"| Machine | {machine.get('chip', '?')}, {_gb(machine.get('ram_bytes'))} RAM |"
    )
    add(
        "| Metal working set | "
        f"{_gb(machine.get('max_recommended_working_set_size'))} |"
    )
    add(
        f"| Packages | vllm {env.get('vllm', '?')}, "
        f"mlx {env.get('mlx', '?')}, "
        f"xgrammar {env.get('xgrammar', '?')} |"
    )
    add(
        f"| Max tokens / repeats | {first['config']['max_tokens']} / "
        f"{first['config']['repeats']} |"
    )
    add(f"| `disable_any_whitespace` | {first['config']['disable_any_whitespace']} |")
    add(
        "| K (`num_speculative_tokens`) | "
        f"{max((r['config']['num_speculative_tokens'] or 0) for r in records)} |"
    )
    add("")
    add(
        "`disable_any_whitespace` is load-bearing, not cosmetic. With free "
        "whitespace allowed between JSON tokens the grammar forces nothing -- "
        "measured coverage falls from 43% to 10% and every draft collapses to a "
        "single token."
    )
    add("")

    add("## Latency")
    add("")
    add("| Batch | Mode | TTFT | TPOT | Output tok/s | Wall | MLX peak |")
    add("|---|---|---|---|---|---|---|")
    for record in records:
        latency = record["latency"]
        full = latency["full_pass"]
        add(
            f"| {record['config']['batch_size']} | {record['mode']} | "
            f"{_ms(latency['ttft_s'])} | {_ms(latency['tpot_s'])} | "
            f"{full['mean_output_tokens_per_s']:.1f} | "
            f"{full['mean_elapsed_s']:.3f} s | "
            f"{_gb(record['memory']['after_decode']['peak_bytes'])} |"
        )
    add("")

    add("### Speedup")
    add("")
    add("| Batch | Arm | TPOT | vs baseline | Verdict |")
    add("|---|---|---|---|---|")
    by_key = {(r["mode"], r["config"]["batch_size"]): r for r in records}
    for batch in sorted({r["config"]["batch_size"] for r in records}):
        base = by_key.get(("baseline", batch))
        if not base or not base["latency"]["tpot_s"]:
            continue
        b_tpot = base["latency"]["tpot_s"]
        add(f"| {batch} | baseline | {_ms(b_tpot)} | 1.00x | — |")
        for arm in COMPARED_ARMS:
            spec = by_key.get((arm, batch))
            if not spec or not spec["latency"]["tpot_s"]:
                continue
            s_tpot = spec["latency"]["tpot_s"]
            ratio = b_tpot / s_tpot
            verdict = (
                "**WIN**" if ratio > 1.02 else ("loss" if ratio < 0.98 else "flat")
            )
            add(f"| {batch} | {arm} | {_ms(s_tpot)} | **{ratio:.2f}x** | {verdict} |")
    add("")

    add("## Speculation quality")
    add("")
    add(
        "| Batch | Arm | Step coverage | Tokens/draft | Tokens from drafts | "
        "Acceptance | #MAT | Altered | Truncated | Rejected |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for record in records:
        if record["mode"] not in PROPOSER_ARMS:
            continue
        cov = _coverage(record)
        if not cov.get("available"):
            add(
                f"| {record['config']['batch_size']} | {record['mode']} | n/a "
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        add(
            f"| {record['config']['batch_size']} | {record['mode']} | "
            f"{cov['coverage'] * 100:.0f}% ({cov['steps_drafted']}/"
            f"{cov['steps_eligible']}) | {cov['tokens_per_draft']:.2f} | "
            f"**{cov['token_share'] * 100:.0f}%** | "
            f"{cov['acceptance'] * 100:.1f}% ({cov['accepted']}/{cov['offered']}) | "
            f"{cov['mat']:.2f} | {cov['altered']} | {cov['truncated']} | "
            f"{cov['rejected']} |"
        )
    add("")
    add(
        "On the `toolspec` row these counters describe the **retrieval half "
        "only** -- its grammar half keeps its own, nested under "
        "`drafter.cumulative.grammar_half` in the JSON. Retrieval acceptance is "
        "expected to be far lower than the grammar's: a grammar draft is legal "
        "by construction, a retrieved one is a guess. The arm's speedup is what "
        "the two produce together.\n\n"
        "*Step coverage* is the share of decode steps that drafted anything; "
        "*tokens from drafts* is the share of emitted tokens that came from a "
        "draft rather than their own forward. The second is the number that "
        "tracks the speedup, and it is much the larger of the two, because a "
        "step that drafts at all carries several tokens. *#MAT* is ToolSpec's "
        "metric -- mean tokens committed per decode step, bonus token included, "
        "so 1.00 is plain autoregressive decoding."
    )
    add("")
    add(
        "**Altered must be zero.** It counts drafts that came back with their "
        "prefix rewritten, which nothing benign does -- it is the unambiguous "
        "evidence that the worker's matcher and the engine's have drifted, and "
        "the only one available, because the synchronous draft handoff edits "
        "drafts silently rather than raising. **Truncated** counts drafts that "
        "came back shorter with the prefix intact; that has two causes the "
        "worker cannot tell apart (the engine's grammar rejecting the tail, or "
        "the scheduler clipping against the token budget / `max_model_len`), so "
        "it is a diagnostic rather than an invariant. **Rejected** is an "
        "ordinary miss: several tokenizations of a forced string are legal, so "
        "a drafted token is grammar-legal by construction but only empirically "
        "the target's argmax."
    )
    add("")

    add("### Cross-check against vLLM's own counters")
    add("")
    add(
        "| Batch | Probe offered/accepted | Prometheus offered/accepted | "
        "Expected ratio | Observed | Agree |"
    )
    add("|---|---|---|---|---|---|")
    for record in records:
        if record["mode"] not in PROPOSER_ARMS:
            continue
        cov = _coverage(record)
        prom = record["drafter"].get("prometheus", {})
        p_off = prom.get("num_draft_tokens")
        p_acc = prom.get("num_accepted_tokens")
        config = record["config"]
        repeats = config.get("repeats") or 0
        warmup = config.get("warmup") or 0
        # The probe counts the measured repeats; Prometheus is cumulative over
        # the whole process, so it also carries the warmup passes. That makes
        # the relationship exact rather than one-sided: probe/Prometheus should
        # be repeats/(repeats+warmup) on both totals.
        expected = repeats / (repeats + warmup) if repeats + warmup else None
        observed = (cov["offered"] / p_off) if cov.get("available") and p_off else None
        agree = "n/a"
        if expected is not None and observed is not None:
            agree = "yes" if abs(observed - expected) <= 0.02 else "**NO**"
        add(
            f"| {config['batch_size']} | "
            f"{cov.get('offered', '?')}/{cov.get('accepted', '?')} | "
            f"{p_off}/{p_acc} | "
            f"{'n/a' if expected is None else f'{expected:.3f}'} | "
            f"{'n/a' if observed is None else f'{observed:.3f}'} | {agree} |"
        )
    add("")
    add(
        "The probe counts the measured repeats and Prometheus counts the whole "
        "process, so the two are expected to differ by exactly the warmup "
        "passes: `repeats / (repeats + warmup)`. Checking that ratio -- rather "
        "than only that Prometheus is the larger -- is what makes this a real "
        "cross-check of the proposer's own accounting."
    )
    add("")

    add("## Cost of drafting")
    add("")
    add("| Batch | propose() | % of wall | Draft walk | Compiles | Vocab table |")
    add("|---|---|---|---|---|---|")
    for record in records:
        if record["mode"] not in PROPOSER_ARMS:
            continue
        cumulative = record["drafter"].get("cumulative", {})
        if not cumulative.get("available"):
            continue
        wall = (
            record["latency"]["full_pass"]["mean_elapsed_s"]
            * (record["config"]["repeats"])
        )
        propose = cumulative.get("propose_seconds", 0.0)
        add(
            f"| {record['config']['batch_size']} | {propose * 1e3:.1f} ms | "
            f"{_pct(propose, wall)} | "
            f"{cumulative.get('draft_seconds', 0.0) * 1e3:.1f} ms | "
            f"{cumulative.get('compiles', 0)} in "
            f"{cumulative.get('compile_seconds', 0.0) * 1e3:.1f} ms | "
            f"{cumulative.get('vocab_table_seconds', 0.0) * 1e3:.0f} ms |"
        )
    add("")
    add(
        "The vocabulary prefix table is built once at drafter construction, not "
        "per step. Grammar compiles are cached by the xgrammar backend, so only "
        "the first request per distinct schema pays one -- on the decode thread, "
        "unlike the engine's threadpool compile."
    )
    add("")

    add("## Correctness")
    add("")
    add("| Batch | Arm | Token ids identical to baseline | Schema-valid outputs |")
    add("|---|---|---|---|")
    for batch in sorted({r["config"]["batch_size"] for r in records}):
        base = by_key.get(("baseline", batch))
        if not base:
            continue
        base_ids = [o["token_ids"] for o in base["latency"]["full_pass"]["outputs"]]
        for arm in COMPARED_ARMS:
            spec = by_key.get((arm, batch))
            if not spec:
                continue
            spec_ids = [o["token_ids"] for o in spec["latency"]["full_pass"]["outputs"]]
            same = "**yes**" if base_ids == spec_ids else "**NO**"
            n = len(spec["latency"]["full_pass"]["outputs"])
            add(f"| {batch} | {arm} | {same} | {spec['schema_valid_outputs']}/{n} |")
    add("")
    add(
        "Speculative decoding is lossless under greedy sampling *in exact "
        "arithmetic*. In floating point it is not bit-exact: packing K+1 query "
        "rows into one forward changes reduction order, so a near-tied argmax "
        "can flip. A mismatch is only evidence of a bug if changing the batch "
        "shape **without** speculation leaves the output unchanged."
    )
    add("")

    if notes:
        add("## Notes")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")

    add("## Caveats")
    add("")
    add(
        "- A drafted token is grammar-legal by construction but only empirically "
        "the target's argmax, because several tokenizations of a forced string "
        "are legal. Acceptance is a measured number, not a guarantee."
    )
    add(
        "- Coverage depends entirely on how much of the schema is skeleton. A "
        "schema dominated by free-form string values will show low coverage, "
        "and that is the honest result rather than a misconfiguration."
    )
    add(
        "- Nothing is drafted for plain chat, for free-form values, or for the "
        "model's choice of which tool to call."
    )
    add(
        "- Single machine, single run set. Run-to-run variance across "
        f"`--repeats {first['config']['repeats']}` is in `elapsed_s`."
    )
    add("")
    return "\n".join(lines)


# -- CLI ---------------------------------------------------------------------


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one configuration and write JSON")
    run.add_argument("--model", required=True)
    run.add_argument(
        "--arm",
        choices=("baseline", "ngram", "grammar", "toolspec"),
        default="baseline",
        help="which drafter to run: none, the in-tree n-gram proposer, the "
        "grammar-forced one, or the ToolSpec port (grammar + retrieval)",
    )
    run.add_argument(
        "--grammar",
        dest="arm",
        action="store_const",
        const="grammar",
        help="alias for --arm grammar",
    )
    run.add_argument(
        "--toolspec",
        dest="arm",
        action="store_const",
        const="toolspec",
        help="alias for --arm toolspec",
    )
    run.add_argument("--batch-size", type=_positive_int, default=1)
    run.add_argument(
        "--sequential",
        action="store_true",
        help="issue --num-prompts distinct prompts one at a time instead of one "
        "batch. Required to measure the toolspec arm: retrieval draws on "
        "requests that have already finished, so within a single batch its "
        "memory is empty by construction.",
    )
    run.add_argument(
        "--num-prompts",
        type=_positive_int,
        default=16,
        help="how many distinct prompts to stream in --sequential mode",
    )
    run.add_argument("--max-tokens", type=_positive_int, default=64)
    run.add_argument("--max-model-len", type=_positive_int, default=1024)
    run.add_argument("--max-num-batched-tokens", type=_positive_int, default=None)
    run.add_argument("--num-speculative-tokens", type=_positive_int, default=8)
    run.add_argument("--repeats", type=_positive_int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--dtype", default="auto")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--prompt", action="append", default=None)
    # select_prompts calls .read_text() on this, so it has to arrive as a Path
    # (gemma4_mtp_benchmark.py, which owns the loader, declares it that way).
    run.add_argument(
        "--prompt-file", default=None, type=Path, help="one prompt per line"
    )
    run.add_argument("--schema-file", default=None, help="JSON schema to constrain to")
    run.add_argument(
        "--no-schema",
        action="store_true",
        help="send unconstrained requests (no grammar at all). The control arm: "
        "the proposer must draft nothing and cost nothing on ordinary chat "
        "traffic, which is what the sonnet run measures.",
    )
    run.add_argument("--chat", action="store_true", help="apply the chat template")
    run.add_argument(
        "--no-thinking",
        action="store_true",
        help="pass enable_thinking=False to the chat template (Qwen3)",
    )
    run.add_argument("--ignore-eos", action="store_true")
    run.add_argument("--prefix-caching", action="store_true")
    run.add_argument(
        "--any-whitespace",
        dest="disable_any_whitespace",
        action="store_false",
        help="allow free whitespace in the grammar (collapses coverage; for "
        "measuring that effect deliberately)",
    )
    run.set_defaults(disable_any_whitespace=True)
    run.add_argument("--label", default=None)
    run.add_argument("--output-json", required=True)

    report = sub.add_parser("report", help="render Markdown from run JSONs")
    report.add_argument("inputs", nargs="+")
    report.add_argument("--output-md", required=True)
    report.add_argument("--note", action="append", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        record = run_one(args)
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str))
        latency = record["latency"]
        print(
            f"{record['label']}: ttft={_ms(latency['ttft_s'])} "
            f"tpot={_ms(latency['tpot_s'])} "
            f"tok/s={latency['full_pass']['mean_output_tokens_per_s']:.1f} "
            f"-> {path}"
        )
        return 0

    records = [json.loads(Path(p).read_text()) for p in args.inputs]
    markdown = render_markdown(records, args.note or [])
    out = Path(args.output_md)
    out.write_text(markdown)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
