#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure how much of a structured-output response the grammar already decided.

This answers a question about the *workload*, not about any particular drafting
implementation: when vllm-metal serves a JSON-schema or tool-calling request, what
fraction of the tokens it spends a full forward pass on were never in question?

It runs plain greedy decoding with **no speculative decoding at all**, captures the
token ids the model emits, then replays that trace through a fresh xgrammar matcher
and counts the positions where ``find_jump_forward_string()`` was non-empty — i.e.
where the grammar had already fixed the next text before the model ran.

That count is deliberately measured with xgrammar alone rather than with
``vllm_metal.v1.grammar_proposer``: it is the *upper bound* on what any
grammar-aware drafter could save, independent of how well a given one captures it.
This script imports nothing from ``vllm_metal`` and runs unchanged on a checkout
that has no grammar proposer in it.

Two modes, because they compile to different grammars:

* ``--mode json`` — a JSON schema, as ``structured_outputs.json``.
* ``--mode tools`` — a real tool-call *structural tag* (free prose until a
  ``<tool_call>`` trigger, then constrained JSON), built the way the OpenAI serving
  layer builds it. This is the shape a live tool-calling request actually gets.

Run it:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_METAL_MEMORY_FRACTION=0.8 \
      python tools/grammar_determinism.py --model Qwen/Qwen3-0.6B --mode tools

For the latency half of the picture — how much of end-to-end time is decode at all —
use vLLM's own tool-calling benchmark rather than this script:

    vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
        --endpoint /v1/chat/completions --model Qwen/Qwen3-0.6B \
        --dataset-name hf \
        --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
        --bfcl-categories simple,live_simple,multiple \
        --num-prompts 100 --temperature 0 \
        --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99

Note that BFCL traffic is *not* grammar-constrained out of the box: its tools carry
no ``"strict": true`` and it sends ``tool_choice: "auto"``, which makes
``get_model_structural_tag`` return ``None``. Grammar enforcement for tool calls is
opt-in; ``--mode tools`` here measures the enforced case.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
from typing import Any

DEFAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["get_weather", "get_forecast", "search_web", "send_email"],
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

DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get a multi-day forecast for a location.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["location", "days"],
                "additionalProperties": False,
            },
        },
    },
]

DEFAULT_PROMPTS = [
    "What is the weather in Paris? Use celsius.",
    "What is the weather in Tokyo? Use fahrenheit.",
    "Give me a 5 day forecast for Oslo.",
    "What is the weather in Cairo? Use celsius.",
    "Give me a 3 day forecast for Lima.",
    "What is the weather in Dublin? Use celsius.",
    "Give me a 7 day forecast for Perth.",
    "What is the weather in Quito? Use fahrenheit.",
]


def load_tools(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.tools_file:
        loaded: list[dict[str, Any]] = json.loads(open(args.tools_file).read())
        return loaded
    return DEFAULT_TOOLS


# BFCL writes its schemas in a Python-flavoured dialect rather than JSON Schema.
# Only the type names differ; the structure is already schema-shaped.
_BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array"}


def _normalise_bfcl_schema(node: Any) -> Any:
    """Rewrite a BFCL parameter schema into JSON Schema xgrammar will compile."""
    if isinstance(node, list):
        return [_normalise_bfcl_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            if value == "any":
                # No JSON Schema equivalent; an absent type means "anything",
                # which is what BFCL means by it.
                continue
            out[key] = _BFCL_TYPES.get(value, value)
        else:
            out[key] = _normalise_bfcl_schema(value)
    # An array must say what it holds or xgrammar cannot build a grammar for it.
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {}
    return out


def load_bfcl_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load (prompt, tools) pairs from Berkeley Function-Calling Leaderboard files.

    Each BFCL entry carries its *own* tool set, so unlike the built-in workload
    every entry compiles to a different grammar. Returned entries are raw; the
    caller turns each into a structural tag.

    BFCL is not grammar-constrained as distributed -- its tools carry no
    ``"strict": true`` and it sends ``tool_choice: "auto"`` -- so ``strict`` is
    forced on here. That is the enforced case this measurement is about, and it
    is what a deployment would set to get constrained tool calls at all.
    """
    items: list[dict[str, Any]] = []
    for path in args.bfcl_file:
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            turns = entry.get("question") or []
            if not turns or not turns[0]:
                continue
            content = turns[0][0].get("content")
            functions = entry.get("function") or []
            if not content or not functions:
                continue
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        # `strict` belongs on the function, not the tool wrapper:
                        # get_model_structural_tag returns None if it is set one
                        # level up, and that silently means "no grammar at all".
                        "strict": True,
                        "parameters": _normalise_bfcl_schema(
                            fn.get("parameters") or {"type": "object", "properties": {}}
                        ),
                    },
                }
                for fn in functions
            ]
            items.append({"id": entry.get("id"), "prompt": content, "tools": tools})
    return items


def build_grammar_spec(
    args: argparse.Namespace, raw_tools: list[dict[str, Any]]
) -> tuple[str, str]:
    """Return ``(kind, spec)`` where kind is ``json`` or ``structural_tag``."""
    if args.mode == "json":
        schema = DEFAULT_SCHEMA
        if args.schema_file:
            schema = json.loads(open(args.schema_file).read())
        return "json", json.dumps(schema)

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

    tools = [ChatCompletionToolsParam(**t) for t in raw_tools]
    tag = get_model_structural_tag(
        model=args.structural_tag_model,
        tools=tools,
        tool_choice="auto",
        reasoning=False,
    )
    if tag is None:
        raise SystemExit(
            "No structural tag was produced. A tag requires a parser model that "
            'supports one and `"strict": true` on at least one tool '
            "(see get_model_structural_tag)."
        )
    return "structural_tag", tag.model_dump_json()


def generate_traces(
    args: argparse.Namespace,
    kind: str,
    spec: str,
    raw_tools: list[dict[str, Any]],
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Greedy decode with the grammar applied and no speculative decoding.

    ``items`` selects corpus mode: a list of per-entry dicts each carrying its
    own ``prompt``, ``tools`` and compiled ``spec``. Without it the single
    shared ``kind``/``spec`` is applied to every prompt.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        # Profiling overhead scales with this, and it is what decides whether a
        # mid-size model fits beside its KV cache on a 24 GB machine: left at
        # vLLM's default, Gemma-4 E2B reserves 8.8 GB of overhead against a
        # 19 GB Metal limit and the KV budget goes negative before the run
        # starts. 512 is what the benchmark harness uses for the same reason.
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.batch_size,
        async_scheduling=False,
        enable_prefix_caching=False,
        structured_outputs_config={
            "backend": "xgrammar",
            "disable_any_whitespace": args.disable_any_whitespace,
        },
    )
    tokenizer = llm.get_tokenizer()

    # The tools must go into the prompt, not just into the grammar. A structural
    # tag permits free text until its trigger fires, so a model that was never
    # told the tools exist simply answers in prose, never triggers the tag, and
    # nothing is ever grammar-determined -- a measurement of the harness rather
    # than of the workload. This mirrors what the OpenAI serving layer does.
    def render(prompt_text: str, tools: list[dict[str, Any]] | None) -> str:
        template_kwargs: dict[str, Any] = {}
        if args.no_thinking:
            template_kwargs["enable_thinking"] = False
        if tools is not None:
            template_kwargs["tools"] = tools
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def params_for(item_kind: str, item_spec: str) -> SamplingParams:
        structured = (
            StructuredOutputsParams(json=item_spec)
            if item_kind == "json"
            else StructuredOutputsParams(structural_tag=item_spec)
        )
        return SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, structured_outputs=structured
        )

    if items is not None:
        # Corpus mode: every entry carries its own tools, so it compiles to its
        # own grammar and needs its own SamplingParams. The engine stays loaded
        # across all of them; only the per-request grammar differs.
        prompts = [render(it["prompt"], it["tools"]) for it in items]
        sampling = [params_for("structural_tag", it["spec"]) for it in items]
        specs = [it["spec"] for it in items]
    else:
        prompts = [
            render(p, raw_tools if kind == "structural_tag" else None)
            for p in (args.prompt or DEFAULT_PROMPTS)[: args.num_prompts]
        ]
        sampling = [params_for(kind, spec)] * len(prompts)
        specs = [spec] * len(prompts)

    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    traces = [list(o.outputs[0].token_ids) for o in outputs]
    texts = [o.outputs[0].text for o in outputs]
    return {
        "traces": traces,
        "texts": texts,
        "specs": specs,
        "tokenizer_name": args.model,
    }


def analyse(
    traces: list[list[int]],
    kind: str,
    spec: str,
    model: str,
    *,
    disable_any_whitespace: bool,
    specs: list[str] | None = None,
) -> dict:
    """Replay each trace through a fresh matcher, counting determined positions.

    The replay grammar must be compiled with the *same* whitespace setting the
    traces were generated under. Compile it stricter and the first whitespace
    token the model emitted is rejected, aborting every trace a couple of tokens
    in and reporting a meaningless share.
    """
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    vocab_size = len(tok.get_vocab())
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info, cache_enabled=True)

    # In corpus mode every trace was generated under its own grammar, so the
    # replay grammar has to be per trace. Compiled contexts are cached by spec
    # because entries frequently repeat a tool set.
    ctx_cache: dict[str, Any] = {}

    def context_for(trace_spec: str) -> Any:
        hit = ctx_cache.get(trace_spec)
        if hit is None:
            hit = (
                compiler.compile_json_schema(
                    trace_spec, any_whitespace=not disable_any_whitespace
                )
                if kind == "json"
                else compiler.compile_structural_tag(trace_spec)
            )
            ctx_cache[trace_spec] = hit
        return hit

    if specs is None:
        specs = [spec] * len(traces)

    total = determined = 0
    runs: list[int] = []
    desynced = 0
    for trace, trace_spec in zip(traces, specs, strict=True):
        matcher = xgr.GrammarMatcher(context_for(trace_spec))
        current = 0
        for token_id in trace:
            # A non-empty jump-forward string means the grammar has already fixed
            # the text at this position -- the forward pass about to run cannot
            # change what comes next.
            if matcher.find_jump_forward_string():
                determined += 1
                current += 1
            elif current:
                runs.append(current)
                current = 0
            total += 1
            if not matcher.accept_token(token_id):
                # The emitted token left the grammar (EOS past termination, or a
                # non-constrained tail). Stop this trace rather than report noise.
                desynced += 1
                break
        if current:
            runs.append(current)

    return {
        "emitted_tokens": total,
        "grammar_determined_tokens": determined,
        "determined_share": determined / total if total else 0.0,
        "forced_runs": len(runs),
        "mean_run_length": statistics.fmean(runs) if runs else 0.0,
        "max_run_length": max(runs) if runs else 0,
        "traces_ending_off_grammar": desynced,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("json", "tools"), default="tools")
    parser.add_argument("--schema-file", default=None)
    parser.add_argument("--tools-file", default=None)
    parser.add_argument("--structural-tag-model", default="hermes")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-thinking", action="store_true", default=True)
    parser.add_argument(
        "--any-whitespace",
        dest="disable_any_whitespace",
        action="store_false",
        help="allow free whitespace in the grammar (collapses determinism; for "
        "measuring that effect deliberately)",
    )
    parser.set_defaults(disable_any_whitespace=True)
    parser.add_argument(
        "--bfcl-file",
        action="append",
        default=[],
        help="Berkeley Function-Calling Leaderboard jsonl file (repeatable). "
        "Each entry brings its own tools, so each compiles to its own "
        "grammar -- a real corpus rather than one hand-written schema.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print(
            "note: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for a single-process run",
            flush=True,
        )

    raw_tools = load_tools(args)
    items: list[dict[str, Any]] | None = None
    skipped_uncompilable = 0

    if args.bfcl_file:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        entries = load_bfcl_items(args)
        random.Random(0).shuffle(entries)
        items = []
        for entry in entries:
            if len(items) >= args.num_prompts:
                break
            try:
                tag = get_model_structural_tag(
                    model=args.structural_tag_model,
                    tools=[ChatCompletionToolsParam(**t) for t in entry["tools"]],
                    tool_choice="auto",
                    reasoning=False,
                )
                if tag is None:
                    raise ValueError("no structural tag produced")
                entry["spec"] = tag.model_dump_json()
            except Exception:
                # A schema this corpus contains but xgrammar/vLLM will not build
                # a tag for. Counted and reported rather than silently dropped,
                # since the skip rate is itself a fact about the workload.
                skipped_uncompilable += 1
                continue
            items.append(entry)
        if not items:
            raise SystemExit("no BFCL entries produced a usable structural tag")
        kind, spec = "structural_tag", items[0]["spec"]
    else:
        kind, spec = build_grammar_spec(args, raw_tools)

    generated = generate_traces(args, kind, spec, raw_tools, items)
    stats = analyse(
        generated["traces"],
        kind,
        spec,
        args.model,
        disable_any_whitespace=args.disable_any_whitespace,
        specs=generated["specs"],
    )
    stats["skipped_uncompilable_entries"] = skipped_uncompilable
    if stats["traces_ending_off_grammar"]:
        # A trace that leaves the grammar makes the share meaningless: the
        # remaining tokens were never counted. Loud, because the usual cause is
        # a replay grammar that does not match the one used to generate.
        print(
            f"WARNING: {stats['traces_ending_off_grammar']} of "
            f"{len(generated['traces'])} traces left the grammar mid-replay; "
            "the share below is computed over a truncated prefix.",
            flush=True,
        )

    print()
    print(f"model            {args.model}")
    print(
        f"grammar          {kind} (disable_any_whitespace="
        f"{args.disable_any_whitespace})"
    )
    print(f"responses        {len(generated['traces'])}")
    print()
    print(f"emitted tokens             {stats['emitted_tokens']}")
    print(
        f"grammar-determined         {stats['grammar_determined_tokens']} "
        f"({stats['determined_share'] * 100:.0f}%)"
    )
    print(
        f"forced runs                {stats['forced_runs']} "
        f"(mean {stats['mean_run_length']:.1f}, max {stats['max_run_length']} tokens)"
    )
    if stats["traces_ending_off_grammar"]:
        print(f"traces ending off-grammar  {stats['traces_ending_off_grammar']}")
    print()
    print(
        "Every determined token cost a full forward pass to produce text the "
        "grammar had already fixed. That share is the ceiling on what a "
        "grammar-aware drafter can remove."
    )

    if args.output_json:
        record = {
            "config": vars(args),
            "grammar_kind": kind,
            **stats,
            "texts": generated["texts"],
        }
        with open(args.output_json, "w") as handle:
            json.dump(record, handle, indent=2, default=str)
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
