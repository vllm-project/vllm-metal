<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vllm-metal-dark.svg">
    <img src="docs/assets/vllm-metal-light.svg" alt="vLLM Metal — 在 Apple Silicon GPU 上运行 vLLM" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/vllm-project/vllm-metal/blob/main/README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
| <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/"><b>官方文档</b></a> | <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/recipes/"><b>按 Mac 内存分类的最佳实践配方</b></a> | <a href="https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0"><b>官方博客</b></a> | <a href="https://slack.vllm.ai/"><b>开发者 Slack</b></a> (#hw-metal) |
</p>

vLLM Metal 是一个硬件加速插件，让 vLLM 能够利用 MLX 作为主要计算后端运行在 Apple Silicon Mac 设备上。它将 MLX 与 PyTorch 统一在单一的 Lowering 执行路径之下。

---
*最新动态* 🔥

- [2026/10] MLX 后端现已提供实验性的 [Recursive Flint INT4](https://huggingface.co/RecursiveSI/flint-int4) 支持，包含并发推理服务能力。
- [2026/10] DiffusionGemma 现已支持在 Metal 上进行类似 Jev 的系统 1（System 1）决策，在 M5 Pro 上的决策中位数延迟低于 0.2 秒。详见[结构化决策指南](https://docs.vllm.ai/projects/vllm-metal/en/latest/structured_decisions/)。
- [2026/10] 桌面端推理会与其他应用程序竞争内存资源。休眠模式（Sleep mode）允许空闲的模型服务器将内存归还给这些应用，而无需关停服务。阅读 [Abhigith Neil Abraham 的详细指南](https://abneil.dev/blog/sleep-mode-on-a-mac-with-kv-offloading.html)。
- [2026/10] 通过 [llmman](https://github.com/llmmanorg/llmman) 结合 vLLM Metal 运行本地模型。详见[配置配方](https://docs.vllm.ai/projects/vllm-metal/en/latest/recipes/#or-with-llmman)。
- [2026/09] 针对 vLLM Metal 与其他推理引擎在运行速度、内存占用与输出保真度方面进行评估的学术论文 [SiliconBench](https://arxiv.org/abs/2609.19169) 已被 [NeurIPS 2026 评估与数据集专题（Evaluations and Datasets Track）](https://neurips.cc/virtual/2026/poster/139658) 录用。
- [2026/09] 阅读我们的[官方 vLLM 公告博客](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)，涵盖 Apple Silicon 上的并发服务架构以及 v0.28.0 新特性。
- [2026/08] vLLM Metal 现已利用 M5 NAX 张量单元加速 MHA、GQA 与 MQA 的 Prefill 计算。
- [2026/08] Qwen3.8 现已支持在 Metal 上运行！`mlx-community/Qwen3.8-27B-8bit` 可在单台 Apple Silicon Mac 上提供 27B 混合 SDPA + GDN 线性模型的高效推理服务。
- [2026/04] 我们发布了全新版本 v0.2.0！统一的分页变长（Paged varlen）Metal 算子现已成为默认的注意力（Attention）计算后端。相比 v0.1.0，首字延迟（TTFT）降低 83 倍，吞吐量提升 3.6 倍。

---

## 系统架构 (Architecture)

上游 vLLM 提供 API 服务器、调度器与分页块管理器（Paged block manager）；mlx_lm 提供逐 Token 计算的模型层；vllm-metal 则负责感知请求的注意力路径 —— 包括分页变长（Paged varlen）算子、M5 NAX Prefill 以及投机解码（Speculative decoding）。

![vllm 架构栈中的 vllm-metal](https://raw.githubusercontent.com/vllm-project/vllm-metal/main/docs/assets/architecture.svg)

## 环境要求 (Requirements)

- macOS 15 (Sequoia) 或更高版本，搭载 Apple Silicon 芯片

## 支持的模型 (Supported Models)

vllm-metal 在 Apple Silicon 上支持持续扩展的模型矩阵。查看完整模型支持列表：[docs/supported_models.md](docs/supported_models.md)。

## 安装指引 (Installation)

- **稳定发行版：** 使用 Homebrew 安装，随后无需激活任何虚拟环境即可直接运行 `vllm`。

  ```bash
  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
  brew install vllm-project/vllm-metal/vllm-metal
  ```

- **最新开发构建版本：** 使用 curl 安装脚本，随后激活生成的虚拟环境。无需本地编译工具链。

  ```bash
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  source ~/.venv-vllm-metal/bin/activate
  ```

- **贡献 Python 或算子核心代码：** 遵循[源码配置指南](docs/CONTRIBUTING.md#development-setup)获取可编辑安装与原生构建工具链。

---

> **免责声明 / Disclaimer**：本文档由社区志愿者维护，致力于提供准确的中文本地化参考。如遇内容更新滞后或歧义，请以 [README.md](https://github.com/vllm-project/vllm-metal/blob/main/README.md) 英文原文为准。
