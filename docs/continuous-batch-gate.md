# Continuous-arrival GDN state-budget gate

`tools/continuous_batch_gate.py` tests real scheduling with heterogeneous input
and output lengths, delayed arrivals, request cancellation, changing batch
membership/order, and KV-pressure preemption/recovery. It complements the
fixed-cohort [state-budget matrix](state-cache-budget-benchmark.md).

Use the same local model, checkout, dependency environment and command options
for each pair. Run the two arms serially. The baseline omits the state budget;
the candidate supplies it. The comparison rejects missing requests, incomplete
outputs, unobserved required events, mismatched identities, and incorrectly
enabled budget roles. It never waives token differences because of a tie.

## Mixed arrivals, multiple concurrency levels

For Qwen3.5-0.8B, the aligned block size is 544; for the tested Qwen3.8-27B-4bit
layout it is 784. The collector checks the resolved size and fails on a mismatch.
Set the model path and use the dedicated environment described in the linked
setup guide. For the 0.8B model:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export HF_HUB_OFFLINE=1
mkdir -p reports/continuous

# Repeat this pair with --max-num-seqs 1, 2, and 4.
python tools/continuous_batch_gate.py run \
  --model /absolute/path/to/local/0.8B-snapshot \
  --scenario mixed --block-size 544 --max-num-seqs 4 \
  --gpu-memory-utilization 0.25 --waves 3 \
  --output reports/continuous/baseline.json

python tools/continuous_batch_gate.py run \
  --model /absolute/path/to/local/0.8B-snapshot \
  --scenario mixed --block-size 544 --max-num-seqs 4 \
  --gpu-memory-utilization 0.25 --waves 3 --state-budget-mib 256 \
  --output reports/continuous/budget.json

python tools/continuous_batch_gate.py compare \
  --baseline reports/continuous/baseline.json \
  --budget reports/continuous/budget.json \
  --output reports/continuous/comparison.json
```

Use separate output filenames for every configuration. For 27B, use its local
snapshot, block size 784, utilization 0.70 and state budget 2048 MiB. Keep the
wave count equal across the concurrency points being compared on one model.

Each mixed wave contains twelve requests with different prompt/output lengths.
One is cancelled during prefill and one after decode output begins. Two other
requests generate past a state-block boundary. New waves arrive every two
seconds by default. Planned and actual submission times are recorded separately:
the in-process driver submits due requests at engine-step boundaries, so it
does not promise exact wall-clock arrival during a blocking step.

The per-request prefill threshold is one aligned block and the total batch
token limit is four blocks. A short request can finish prefill while an older,
longer request is still in prefill. The runner's ordinary decode-before-prefill
packing can therefore change their relative positions without the test
reordering the scheduler itself. A concurrency-one run cannot cover mixed
worker batches or relative reordering; its requirements explicitly reflect that.

## Real KV-pressure preemption and recovery

Use an explicit small KV pool rather than calling a preemption method from the
test. For the tested block layouts, 48 total blocks leave enough room for one
request to finish, while concurrent requests can exhaust available KV blocks.
The gate requires actual preempted and later resumed IDs, followed by normal
completion. Queuing, prefix-cache restoration and waiting for streamed input do
not count as preemption.

Run another baseline/budget pair with the same model-specific settings, changing
both arms to:

```text
--scenario pressure --max-num-seqs 4 --num-gpu-blocks-override 48 --waves 1
```

Each pressure wave contains eight requests. The explicit override is a stress
configuration, not evidence that the natural startup planner chooses 48 blocks.
It limits the scheduler's logical shared BlockPool. The Metal backend may retain
larger KV arrays allocated during startup profiling; report both
`resolved.num_gpu_blocks` and `runtime_before.num_blocks`. This exercises real
scheduler allocation failure and preemption, not operating-system memory
exhaustion or a claim that physical KV storage shrank to 48 blocks.
Do not reduce it below the model's startup capacity requirement. If the selected
workload does not actually preempt and recover a request, the gate fails instead
of silently marking that scenario covered.

## State-budget behavior under pressure

The opt-in budget preserves the existing deferred-free fence: blocks referenced
by an unfinished GPU step stay owned until its result is consumed. Before core
scheduling, the budget adapter bounds the new blocks needed by running requests
from their existing tables. It first releases eligible states strictly before
the completed source. If global blocks or unpinned state capacity are still
insufficient, it dispatches a normal zero-token step so the existing FIFO can
consume earlier work. Once no positive-token step is outstanding, ordinary
preemption can free blocks immediately. Continuing within an existing decode
block needs no new block and does not trigger this wait.

`pressure_wait_steps`, `pressure_wait_global_steps` and
`pressure_wait_state_steps` count these waits; the reason counts can overlap.
They are separate from quota/admission allocation failures. The supported
manager geometry is ordinary full attention plus aligned Mamba state, with
existing exclusions for partial CoW, internal checkpoints, speculation and
external transfers. Unknown manager implementations are rejected explicitly.

Waiting changes batch composition and which computations finish before a
preemption. Keep complete-output compatibility against the original production
schedule separate from any same-schedule state/reference experiment. A reference
match does not rewrite a failed original comparison as PASS. Preserve the old
report and identify every deliberately changed scheduling policy in diagnostics.

## What is measured

The collector uses the real in-process `LLMEngine`, including its resolved
asynchronous scheduler and decode pipeline. Instance-local observers forward
all original arguments unchanged and read CPU bookkeeping. They record actual
runner order, state-slot mappings, allocation/quota metadata, scheduler
preempt/resume IDs, and engine-confirmed cancellation. Observers do not evaluate
GPU state, synchronize per step, add logprobs or manufacture ordering changes.

Complete non-cancelled outputs and finish reasons are compared by planned
request ID. Cancelled requests retain their entire delivered prefixes and are
reported separately. Every planned request must appear, even if it fails or is
cancelled. A client calling abort is insufficient without engine confirmation.

Performance output includes per-request first-token and completion latency,
per-token **delivery intervals**, p50/p95/p99 with sample counts, and a throughput
window from first submission to last termination. Multiple tokens returned in
one callback share its timestamp; these observations are not GPU kernel timing.
Arrival gaps, queueing, prefill and cancellation work are included in the window;
engine startup is separate. Memory statistics retain MLX allocator and RSS
definitions from the fixed-cohort tool.

These finite workloads establish their observed event coverage and output
comparisons. They are not an HTTP/network test, a multiprocessing-shutdown test,
or a long-duration service soak. All ordinary requests currently use fixed
output limits with `ignore_eos=True`; varied limits and explicit cancellation
exercise departures, but do not constitute natural-EOS coverage. For that
missing check, run a **separate** pair with `--no-ignore-eos --apply-chat-template`
so prompts go through the model's chat template and sequences may finish on
configured EOS tokens. Keep the same non-budget arguments in both arms. Record
finish reasons and complete outputs; do not relabel a fixed-length
`ignore_eos=True` archive as natural-stop evidence. Matching-schedule
no-quota/C83 diagnostic controls are host-specific: unless a host re-runs
them, treat the published controls as belonging to the original 64 GB archive.
