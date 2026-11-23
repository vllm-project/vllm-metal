# Release process

1. Update `[project].version` in `pyproject.toml` and `.github/vllm-release-tag.commit` on `main`.
2. Cut the branch: `git switch -c releases/vX.Y.Z main && git push origin releases/vX.Y.Z`.
3. Run **Build and Release** with `X.Y.Z`; publish `X.Y.ZrcN` first only when an RC is useful.
4. Confirm the draft contains both Python 3.12 (`cp312`) and 3.13 (`cp313`) wheels. Test each with representative attention types, including Qwen3 (GQA) and Qwen3.5 (hybrid SDPA + GDN), plus speculative decoding.
5. Review the generated notes, edit the draft, and publish it.
6. For a later fix, cherry-pick it to the branch and release `X.Y.Z.post1`.
7. For a stable release, **Update Homebrew** opens a PR to update `Formula/vllm-metal.rb`. Review and merge that PR.
