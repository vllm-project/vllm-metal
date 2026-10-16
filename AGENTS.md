# Agent instructions for vllm-metal

Follow [the contributor guide](docs/CONTRIBUTING.md) for setup and general checks.
For Omni additions, follow the [model ownership and layout](docs/pytorch-mps.md#adding-omni-models).

## Motivation

- Address real problems backed by a reproducer, measurements, or a concrete user need. Avoid fixes justified only by hypothetical failures.
- State the motivation clearly in the PR description: the current behavior, the affected use case, and why the issue matters in practice.

## Duplicate-work checks

Before opening a PR, search open PRs by topic. When working from an issue, read its comments and search for PRs referencing it:

```bash
gh pr list --repo vllm-project/vllm-metal --state open --search "<short area keywords>"
gh issue view <issue_number> --repo vllm-project/vllm-metal --comments
gh pr list --repo vllm-project/vllm-metal --state open --search "<issue_number> in:body"
```

Do not open a duplicate PR. Explain any materially different approach in the PR description.

## Coding style

- Match existing code style.
- Keep changes focused; avoid unrelated refactors and cleanups.
- Reuse existing utilities. Add helpers or abstraction layers only when they serve a concrete current need.
- Keep one-off experiments, debug harnesses, and low-value tests local, outside the PR. Include durable regression tests and necessary fixtures.
- Validate inputs at boundaries. Keep unexpected failures visible; avoid broad exception handling or fallback values that conceal broken internal contracts.
- Prefer legible, self-documenting code. Remove redundant comments. Use comments for non-obvious intent or constraints, and keep comments and docstrings brief and direct.

## Validation

Run checks relevant to the change. Report the commands, results, and any checks you could not run in the PR description.

Extend related test suites. Test observable behavior rather than locking in implementation details.

### Test design

When adding tests:

- **Design before you write.** Answer four questions first: what is the module for, what is its I/O contract, what failure am I guarding against, and what is the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs; state why in the name or docstring. Skip trivial wiring; flaky tests are worse than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in `tools/benchmark/`; prove correctness in existing pytest suites.

### Model parity and benchmarks

For model changes, use parity tools with matching checkpoints and input token IDs.

Always set `PYTORCH_ENABLE_MPS_FALLBACK=0` for MPS runs to disable PyTorch's optional CPU fallback.

- **MLX and PyTorch MPS:** compare top-K parity against `mlx_lm` using [the parity tool](tools/check_parity.py) with `--top-k 5`. Set `VLLM_METAL_BACKEND=mps` to check PyTorch MPS serving. Report `EXACT` and `TOP_K_MATCH` counts separately and investigate failures.

For performance improvements, report before/after end-to-end serving speedup on meaningful workloads using `vllm bench serve`. Follow [the macOS benchmarking guide](docs/benchmarking-macos.md) and keep the model, workload, and serving configuration consistent between runs.

### Speculative decoding

- Compare fresh greedy runs with speculative decoding enabled and disabled, using the same inputs and settings. Accept `EXACT` or mutual `TOP_K_MATCH` at the first divergence (`--top-k 5`); require zero `FAIL`.
- Test the algorithm's correctness properties, including exact verifier results for fixed target logits. For draft-model speculation, use the same checkpoint for draft and target under greedy decoding; acceptance should be near 100%. Report the measured rate and investigate substantial rejections.
- Evaluate acceptance rate and speedup on natural datasets such as [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks), prepared with [the dataset builder](tools/build_spec_bench_dataset.py). Artificial workloads such as `sonnet` can produce unrealistically low acceptance rates.
- Demonstrate end-to-end speedup at batch size `B=1`. Speedup at `B>1` is harder but still expected; benchmark both with [the benchmark tools](tools/README.md). Report the tested batch sizes, representative workloads, and any regressions.

## Commit messages

- Use a short, imperative subject describing the change. Explain the reason in the body when it is not obvious.
- Sign off commits with `git commit -s`. Add `Co-authored-by:` trailers when applicable.
