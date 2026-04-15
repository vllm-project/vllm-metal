# Contributing to vLLM Metal

Thanks for your interest in contributing! This plugin targets **Apple Silicon Macs only** — you'll need an M-series Mac running macOS to build, test, and run it.

## Development setup

```bash
git clone https://github.com/vllm-project/vllm-metal.git
cd vllm-metal

# Creates ./.venv-vllm-metal/ and installs vLLM core + the plugin
./install.sh

# Activate the virtualenv
source .venv-vllm-metal/bin/activate

# Install dev dependencies (pytest, ruff, mypy, ...)
pip install -e ".[dev]"
```

## Run lint locally

Mirrors the `lint` job in CI (`ruff`, `ruff format --check`, `mypy`, `shellcheck`):

```bash
scripts/lint.sh
```

## Run CI locally

Mirrors the `test` job in CI — serving smoke tests plus the non-slow pytest suite:

```bash
scripts/test.sh
```

> For a faster inner loop while iterating, run pytest directly:
>
> ```bash
> pytest -m "not slow" tests/ -v --tb=short
> ```

🎉 **Congratulations!** You have completed the development environment setup.

---

## Before you open the PR

Two conditional checks apply depending on what your PR touches:

**If your PR adds or modifies a model**, include a deterministic test that asserts the generated tokens match the `mlx_lm` reference under greedy sampling (`temperature=0`). See `tools/gen_golden_token_ids_for_deterministics.py` for how to generate golden token IDs for a new model.

**If your PR claims a performance improvement**, attach before/after benchmark results. For example, using `vllm bench serve` with the sonnet dataset:

```bash
curl -O https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt

# 1. Start the server
VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.8 \
  vllm serve Qwen/Qwen3-0.6B --port 8000 --max-model-len 2048

# 2. Run the benchmark
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --dataset-name sonnet \
  --dataset-path sonnet.txt \
  --num-prompts 100 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,99
```

## Developer Certificate of Origin (DCO)

When contributing changes to this project, you must agree to the [DCO](https://developercertificate.org/). Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Submit your changes

1. **Fork** the repository on GitHub.

2. **Re-point `origin` to your fork and add `upstream`:**

   ```bash
   git remote set-url origin https://github.com/<your-username>/vllm-metal.git
   git remote add upstream https://github.com/vllm-project/vllm-metal.git
   ```

3. **Create a feature branch:**

   ```bash
   git checkout -b my-feature
   ```

4. **Commit your changes using `-s`** (adds the DCO sign-off automatically):

   ```bash
   git commit -sm "your commit info"
   ```

5. **Push to your fork:**

   ```bash
   git push -u origin my-feature
   ```

6. **Open a pull request** against `main` in the upstream repository.
