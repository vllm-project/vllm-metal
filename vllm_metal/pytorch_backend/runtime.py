# SPDX-License-Identifier: Apache-2.0
"""Device adapters for vLLM 0.32's upstream MRV2 runner.

PyTorch's native streams and events use the single MPS default queue. Staging
buffers use explicit CPU-to-MPS copies.
"""

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


def compute_topk_scores(
    logits,
    num_logprobs,
    sampled_token_ids,
    cu_num_logits=None,
    logprob_token_ids_state=None,
    expanded_idx_mapping=None,
    max_per_req_token_ids=0,
    logits_mode=False,
):
    from vllm.v1.sample.sampler import Sampler

    # Explicit-token and prompt logprobs remain rejected by validate_request.
    assert max_per_req_token_ids == 0
    scores = logits.float() if logits_mode else Sampler.compute_logprobs(logits)
    # Keep upstream's compiled rank helper eager on MPS.
    with torch.compiler.set_stance("force_eager"):
        result = Sampler.gather_logprobs(scores, num_logprobs, sampled_token_ids.long())
    if isinstance(cu_num_logits, torch.Tensor):
        return result._replace(cu_num_generated_tokens_tensor=cu_num_logits)
    return result._replace(cu_num_generated_tokens=cu_num_logits)
