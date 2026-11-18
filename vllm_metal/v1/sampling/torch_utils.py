# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm b46907cce0f7 (== v0.33.0), Apache-2.0:
# make_tensor_with_pad from vllm/utils/torch_utils.py, which
# vllm-project/vllm#63373 removes (make_ndarray_with_pad stays upstream).
# Local modifications: a PEP 695 type parameter replaces the module TypeVar.
"""Tensor helpers the vendored sampling stack needs."""

import torch
from vllm.utils.torch_utils import TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad


def make_tensor_with_pad[T](
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor
