# Nemotron Nano VL V2

Nano VL V2 accepts images and produces text through the paged Nemotron-H
runtime. The model-owned adapter runs MLX-VLM's vision encoder and projector,
then passes their output to the existing text runtime. Audio, video, and
precomputed image embeddings receive validation errors. Omni support requires
its own adapter and processor contract.

The October 27 concurrency repair passes the bounded checkpoint, HTTP, and
async lifecycle checks with MLX 0.32.4 and MLX-VLM 0.7.6. The
[validation record below](#october-27-validation-and-concurrency-repair) pins
the image implementation and the combined runtime source, which includes
separate memory controls. The tested upstream base is
`cac610b74b84b5b4069dab88148b7b9a49200aee`.

## Checkpoint and preprocessing

The reference checkpoint is `ekryski/Nemotron-Nano-12B-v2-VL-4bit`, revision
`e47ef857c31650d807610d810be199a20d9bf114`. October 27 strict loading consumed 1,037 tensors:
641 language tensors, 389 vision tensors, and seven projector tensors. No
checkpoint tensors are discarded and no model parameters are left unfilled.

MLX-VLM 0.7.6 calls the matching model implementation `nemotron_h_nano_omni`.
The compatibility layer maps the V2 model type and processor without changing
checkpoint files. Its sanitizer removes an unshipped video projection only
for V2 checkpoints; incomplete unrelated Omni checkpoints still fail strict
loading, and a shipped video projection is retained. A local configuration
reader handles MLX exports that omit NVIDIA's Python config files, without
requiring `trust_remote_code`.

The loader mapping, current MLX-VLM interfaces, encoder profiling, and cached
tensor copy handling incorporate ideas from syang-genai's
[PR #1160](https://github.com/vllm-project/vllm-metal/pull/1160), commit
`96ba69519bd7c3d8ce83591fa31c33b3a29223d7`.

Image preparation follows NVIDIA's static VL V2 processor, pinned at
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` revision
`ca9543b126e8bf3176916d3d305ccc415f89fd4d`. Pixels are rescaled before bicubic
tensor resizing, split into 512-pixel tiles, and followed by a thumbnail when
there is more than one tile. Normalization precedes conversion to BF16.
Each tile produces 256 image embeddings. The `<img>` and `</img>` markers stay
as text embeddings; only their enclosed image positions receive visual rows,
including when a prefill chunk divides that range.

The generic Omni processor uses different image geometry. Successful loading
through the Omni model class does not make that processor a VL V2 reference.
Dynamic-resolution configurations are rejected by this image adapter.

Encoder profiling follows the server's configured `max_num_tiles`, including
the thumbnail. Per-request overrides cannot exceed that ceiling. Disabling
image inputs also disables encoder profiling. Processor pixel precision is
preserved, including when projector weights use packed quantized storage.

## Bounded serving example

Use a local copy of the pinned checkpoint with:

```bash
MLX_ENABLE_TF32=1 \
  vllm serve /path/to/Nemotron-Nano-12B-v2-VL-4bit \
  --host 127.0.0.1 --served-model-name nano-vl-v2 \
  --max-model-len 2048 --max-num-seqs 2 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.5 \
  --no-enable-prefix-caching --enforce-eager --generation-config vllm \
  --limit-mm-per-prompt '{"image":2}' \
  --mm-processor-kwargs '{"max_num_tiles":2}'
```

Send standard `/v1/chat/completions` messages containing `text` and `image_url`
content parts. Images can use data URLs. For reproducible greedy comparisons,
set `temperature=0`, `seed=0`, and `max_tokens=32`. vLLM's `return_token_ids=true`
request extension exposes the prompt and generated IDs for comparison.

Two tiles plus a thumbnail consume 768 image positions. The tile and image
limits must fit the complete text-and-image context. These settings describe
the intended bounded workload. They are not a performance or headroom claim.
The local validation harness additionally uses separately maintained explicit
KV-byte and wired-limit controls; those changes are not part of this image patch.

## October 27 validation and concurrency repair

The first October 27 run passed all eight sequential cases but changed the
first token of the two-image request in a concurrent pair. Its prompt IDs and
input embeddings matched the direct reference. The first differing layer was
an attention layer: its quantized BF16 K/V projections changed when the
request's final 51-token prefill chunk shared a larger packed step with another
request. Subsequent recurrent layers amplified the difference. The fix projects
Nano K/V separately for each request, preserving the standalone projection
geometry while retaining paged attention and concurrent scheduling.

Image implementation `77dc330354087357d743e571eac63a3c7a9a8735` contains the
repair and its regression test. The actual runtime checks used combined source
`3b7b69c817d814a7fe30cc1e7dadd19422972e70`, which includes the same image code
plus the separately maintained explicit cache-byte and wired-memory controls.
Those controls are not part of the image diff. The cache-byte change has since
merged through [PR #803](https://github.com/vllm-project/vllm-metal/pull/803);
the wired-memory option remains a local companion.

| Check | Result |
| --- | --- |
| Strict reference | All 1,037 tensors loaded; none ignored, stripped, unconsumed, or missing |
| Reference verification | Eight fresh capsules and their tensor inventory passed schema and self-digest checks |
| Paged parity | Exact prompt and output IDs, including EOS, for eight sequential cases and three concurrent pairs |
| Independent checkpoint suite | 11 passed; its fresh reference also matched all eight validated capsules |
| Normal HTTP server | 12 checks passed; clean shutdown with exit code zero |
| Async lifecycle | Six checks passed: abort, generator close, audio/video/image-embedding rejection with recovery, and concurrency after cancellation |
| Focused offline suites | 157 passed on the combined source, including the real quantized K/V regression |

The real-projection regression failed for both unequal prefill orders before
the fix and passes afterward, including single-token decode rows. Targeted
Ruff/format checks and mypy over 188 runtime files passed. The reference's
incorrect `orange, blue` answer remains unchanged: exact token parity measures
equivalent execution, not image-understanding quality.

The machine was an Apple M5 Pro with 20 GPU cores and 64 GB memory, running
macOS 26.6.2. Dependencies were vLLM 0.32.0+cpu, Torch 2.13.0, Transformers
5.19.0, MLX 0.32.4, and MLX-VLM 0.7.6. The same 25 native source files and
nine reused wheel assets were verified; this was not a native rebuild.

The run retained the serving bounds above, seed zero, 32 output tokens,
`MLX_ENABLE_TF32=1`, and `PYTORCH_ENABLE_MPS_FALLBACK=0`. The combined source
additionally used `--kv-cache-memory-bytes 1073741824` and
`VLLM_METAL_DISABLE_WIRED_LIMIT=1`. All five stages passed their 900-second
process-group guards and all ten host boundaries passed with zero swap on the
same approved boot. The scoped host exception covered only an absent historical
panic log. An earlier diagnostic cleanup error and its passing recovery check
remain recorded separately; no final acceptance stage had a cleanup error.
These results do not measure projection overhead, endurance, or higher concurrency.

## Historical trained evidence

The September direct reference used MLX-VLM 0.6.8, MLX 0.32.1, and NVIDIA's
independently loaded image processor on an Apple M5 Pro with 64 GB memory,
macOS 26.6.2, and Python 3.12.12. The reference uses the loader adjustment
described above, but does not call the Metal adapter, Metal processor, or
paged runtime. Each case uses a fresh hybrid cache and greedy decoding.
The reference explicitly pins `MLX_ENABLE_TF32=1`, MLX's default math mode.
Changing this setting can change exact greedy tokens on M5: with TF32 disabled,
one tiled case produced `Red` where the pinned reference produced `red`.
The parity test starts a fresh reference process and engine with the same setting,
because MLX reads it once per process. It restores the surrounding test
environment afterward; ordinary unit tests keep their strict FP32 setting.

Paged Python serving reproduced every prompt ID and generated ID, including
EOS, for eight cases: text only, square, resized, wide tiled, portrait tiled,
two images, reversed images, and multi-turn image history. Three pairs also
matched with two simultaneous requests. Separate async checks exercised
explicit cancellation, generator closure, recovery, request-state cleanup,
and rejection of audio, video, and precomputed image embeddings.

Normal HTTP serving reproduced all eight cases in one uninterrupted server
session, including prompt/output IDs and usage accounting. The same run
verified streaming token IDs including EOS, usage and terminal events, client
disconnect/recovery, structured HTTP 400 responses for unsupported audio and
excess images, and recovery after both errors. All 12 checks passed using the
normal CLI and enabled adapter, with clean startup and shutdown. Earlier
interrupted attempts remain recorded separately in the evidence.

These results were recorded on September 14, 2026, against upstream base
`bc9bd16f8156e38f6e2c7143d6964090b2dcaf8b`. One reference answer called the
red image "orange" in a two-image prompt. Token parity measures equivalent
execution, not visual accuracy.

On September 16, the strict reference, all 11 fresh-reference checkpoint tests,
and all 12 HTTP checks passed again after refreshing onto upstream
`176bad62480f028baee714292925bffabb2ef50b`. The reference consumed all 1,037
tensors and preserved every earlier prompt and output token. HTTP serving
shut down with exit code zero. The standalone async lifecycle results above
remain from September 14.

## Reproducing the checkpoint checks

The synthetic image recipes, messages, checkpoint hashes, and reference pins
are in [`tests/fixtures/nano_vl_v2_cases.json`](../tests/fixtures/nano_vl_v2_cases.json).
There are no frozen expected output tokens. The opt-in test generates a fresh
direct MLX-VLM reference for all eight cases, exits that process, and then
compares the paged engine with it, including three concurrent pairs. Every
prompt ID and generated ID must match exactly, including EOS; a top-K
alternative is not accepted. The reference JSON and log remain in pytest's
temporary output directory for inspection.

After checking host headroom and obtaining the local checkpoint, the opt-in
test is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  NEMOTRON_NANO_VL_PARITY_MODEL=/path/to/Nemotron-Nano-12B-v2-VL-4bit \
  NEMOTRON_NANO_VL_PROCESSOR_SOURCE=/path/to/pinned/image_processing.py \
  pytest -q tests/multimodal/test_nemotron_nano_vl_parity.py
```

Obtain `image_processing.py` from the pinned NVIDIA revision above. The test
verifies its SHA-256 before importing it and verifies the checkpoint hashes
before loading. It downloads no weights and checks the normal enabled adapter
without a validation-only override. The reference uses the same input
preparation and greedy loop as the separately recorded capsule and tensor
inventory validation. Its only shared runtime dependency is loader compatibility;
it does not call the Metal adapter or processor.
Run `pytest -m 'not slow' tests/multimodal` for checkpoint-free regression
coverage. Other checkpoints, quantizations, longer contexts, and performance
require separate evidence.
