# Parallelism selector implementation plan

Goal: Pipeline-default, functional Pipeline/Tensor selection in Gradio for the same GPT-OSS checkpoint, vLLM-Metal server and strict JACCL transport.

Architecture: Support GPT-OSS TP2/PP1 in the existing Metal runner, retaining PP2/TP1. Manager reloads backend children while keeping the UI alive; UI admission and manager metrics checks protect active requests. Publish active mode and switch progress separately from requested mode.

Constraints: no TCP model-transport fallback; preserve existing uncommitted work; no combined TP+PP/DP, speculative decoding or LoRA under initial TP support; shared runtime across chat/code-completion; preserve browser history; leave Pipeline active after validation.

## Backend
- [x] Add TP admission/sharding/token synchronization tests, observe failures.
- [x] Add small tensor.py owner for config validation, native GPT-OSS sharding, local KV metadata, authoritative rank-zero token synchronization.
- [x] Bootstrap JACCL for TP workers, lazy-load then shard before weight evaluation, disable local-only optimization probes under TP.
- [x] Validate targeted tests then actual two-Mac loading and inference, synchronize changed backend files to mac-smb.

## Runtime and UI
- [x] Add Pipeline/Tensor launch config tests, mode-specific world sizes/transport and memory budget; default Pipeline.
- [x] Add switch protocol and tests for busy requests, ownership, errors and completion; keep UI process alive during backend reload.
- [x] Add global selector, explicit Apply, active state and progress; protect both chat and completion requests during reload.
- [x] Update README and backend docs.

## Validation
- [x] Test tensor prefill/decode, non-greedy sampling, concurrent chats, cancellation and return to Pipeline.
- [x] Browser-verify selector, reload progress, preserved saved chat and real inference for both modes.
- [x] Run relevant full suites and inspect final diff; leave working Pipeline default service running.

Validation: 2,225 backend non-slow tests passed, 5 skipped, 24 slow deselected; 77 UI tests passed; Ruff and targeted mypy passed. Both modes served real GPT-OSS requests. Tensor tested with cold/cached 2,159-token prefill, random sampling, concurrent requests, UI-triggered reloads and conversation recall. UI reload admission remains closed after cancellation until manager acknowledgement. Pipeline is the final active/default mode.
