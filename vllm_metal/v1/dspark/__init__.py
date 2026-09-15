# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative-decoding drafter for the Metal V1 runner.

An MLX implementation of the DeepSpec DSpark inference path for the Qwen3 and
Gemma4 drafter families; only Qwen3 is enabled for serving. ``config.py``,
``model.py`` and ``loader.py`` derive from ARahim3/mlx-dspark (MIT); see NOTICE
in this package.
"""
