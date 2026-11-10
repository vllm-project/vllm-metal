<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE.

## Contributor intent (required human answers)

All four questions must be answered by the human contributor before submission.
An agent may reuse explicit human answers with a public-safe source reference;
it must not invent, infer, or prefill missing answers. If an answer is missing
or materially ambiguous, ask the human and wait. Do not publish private
conversation content without authorization.

### 1. What outcome do you want to achieve?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 2. What is the highest-leverage variable for that outcome?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 3. How will this work move that variable?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 4. Whose user or workload needs does this address, and what are those needs?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

## Purpose

<!-- Describe the behavior before and after this change. Link the affected
module/feature contract and user guide/example, or explain why unchanged. -->

## Test Plan

**Model / checkpoint:**

**Hardware / macOS:**

**vLLM Metal commit:**

## Test Result

<!-- Commands and actual results at the reported commit. Explicitly list
relevant checks not run. For performance claims, include before/after
vllm bench serve results. -->

**BEFORE SUBMITTING:** read [the contributor guide](https://github.com/vllm-project/vllm-metal/blob/main/docs/CONTRIBUTING.md) and run `scripts/lint.sh` plus the relevant tests.

> [!NOTE]
> To avoid agent spam, vLLM Metal limits contributors without write access to **5 open pull requests**, including drafts. If this blocks well-intentioned critical work, please contact a committer to request exemption-list access.
