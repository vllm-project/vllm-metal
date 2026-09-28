# SPDX-License-Identifier: Apache-2.0
"""Device adapters for vLLM 0.30's upstream MRV2 runner.

PyTorch's native streams and events use the single MPS default queue. Staging
buffers use explicit CPU-to-MPS copies.
"""

from functools import partial

import numpy as np
import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import NonUvaBuffer


class MPSStagingBuffer(NonUvaBuffer):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        # The shared MLX platform advertises CPU as its Torch device.
        self._uva = self._uva.to("mps")

    def uva(self, n=None):
        result = super().uva(n)
        result._metal_cpu = self.cpu if n is None else self.cpu[:n]
        return result


def copy_to_device(x, **kwargs):
    result = async_tensor_h2d(x, **kwargs)
    result._metal_cpu = torch.as_tensor(x)
    return result


def cpu_mirror(tensor):
    mirror = getattr(tensor, "_metal_cpu", None)
    if mirror is not None:
        return mirror
    base = tensor
    while base._base is not None:
        base = base._base
    mirror = getattr(base, "_metal_cpu", None)
    if mirror is None:
        return tensor.cpu()
    return mirror.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())


def copy_changed_state(self, n=None):
    source = self.np if n is None else self.np[:n]
    snapshot = getattr(self, "_metal_snapshot", None)
    if snapshot is None or not np.array_equal(source, snapshot):
        self.gpu = self.pool.copy_to_uva(source)
        self._metal_snapshot = source.copy()
    return self.gpu


def staged_write(self):
    if not self._staged_write_indices:
        return
    offsets = []
    start = 0
    stride = self.gpu.stride(0) if self.gpu.ndim > 1 else 1
    for row, col, end in zip(
        self._staged_write_indices,
        self._staged_write_starts,
        self._staged_write_cu_lens,
        strict=True,
    ):
        offsets.extend(range(row * stride + col, row * stride + col + end - start))
        start = end
    indices = torch.tensor(offsets, device=self.device, dtype=torch.int64)
    values = torch.tensor(
        self._staged_write_contents, device=self.device, dtype=self.dtype
    )
    self.gpu.view(-1).index_copy_(0, indices, values)
    self.clear_staged_writes()


def install():
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.v1.worker.gpu import buffer_utils, model_runner
    from vllm.v1.worker.gpu.sample import sampler

    from vllm_metal.pytorch_backend import input_ops

    if getattr(buffer_utils, "_metal_installed", False):
        return

    @ShortConv.register_oot
    class MPSShortConv(ShortConv):
        # Reuse upstream Torch ops without its CPU/CUDA custom-op wrapper.
        forward = ShortConv.forward_native

    # MRV2 0.30 uses these CUDA names even with an MPS device. Use native
    # PyTorch objects; partial avoids duplicate Dynamo handler registration.
    torch.cuda.Stream = torch.Stream
    torch.cuda.Event = partial(torch.Event, device="mps")
    torch.cuda.current_stream = partial(torch.accelerator.current_stream)
    torch.cuda.set_stream = partial(torch.accelerator.set_stream)
    # Explicit vLLM 0.30 bindings for the supported single-device path.
    buffer_utils.NonUvaBuffer = MPSStagingBuffer
    model_runner.async_tensor_h2d = copy_to_device
    buffer_utils.StagedWriteTensor.apply_write = staged_write
    buffer_utils.UvaBackedTensor.copy_to_uva = copy_changed_state

    input_ops.install()
    # validate_request limits this baseline to plain greedy generation.
    sampler.gumbel_sample = lambda logits, *args, **kwargs: logits.argmax(dim=-1)
    buffer_utils._metal_installed = True
