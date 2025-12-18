# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible async utilities.

This module provides Metal/MLX-compatible implementations of async
utilities that replace CUDA stream-based operations.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MetalAsyncOutput:
    """Async output handler for Metal backend.

    On Metal, we don't have CUDA streams, so this is a simplified
    synchronous implementation. Apple Silicon's unified memory
    architecture makes this less of a bottleneck compared to
    discrete GPU systems.
    """

    tensor: torch.Tensor | None = None
    is_ready: bool = False
    result: Any = None

    def __init__(self, tensor: torch.Tensor | None = None):
        """Initialize async output.

        Args:
            tensor: Optional tensor to wrap.
        """
        self.tensor = tensor
        self.is_ready = tensor is not None
        self.result = None

    def wait(self) -> torch.Tensor | None:
        """Wait for the async operation to complete.

        On Metal, this is a no-op since we use synchronous operations.

        Returns:
            The tensor result.
        """
        # Synchronize MLX if needed
        try:
            import mlx.core as mx

            mx.eval([])
        except ImportError:
            pass

        # Synchronize MPS
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        self.is_ready = True
        return self.tensor

    def get_result(self) -> Any:
        """Get the async result.

        Returns:
            The result, waiting if necessary.
        """
        if not self.is_ready:
            self.wait()
        return self.result

    def set_result(self, result: Any) -> None:
        """Set the result.

        Args:
            result: Result to store.
        """
        self.result = result
        self.is_ready = True

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "MetalAsyncOutput":
        """Create async output from tensor.

        Args:
            tensor: Tensor to wrap.

        Returns:
            MetalAsyncOutput instance.
        """
        return cls(tensor=tensor)


class MetalOutputBuffer:
    """Output buffer for Metal backend.

    This manages output tensors for batch inference, providing
    a similar interface to CUDA-based pinned memory buffers but
    using unified memory instead.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.long,
        device: torch.device | str = "cpu",
    ):
        """Initialize output buffer.

        Args:
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.
            dtype: Data type for the buffer.
            device: Device for the buffer (use CPU for unified memory).
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-allocate buffer
        self.buffer = torch.zeros(
            (max_batch_size, max_seq_len),
            dtype=dtype,
            device=self.device,
        )

        # Track usage
        self.current_batch_size = 0
        self.current_seq_len = 0

    def reset(self) -> None:
        """Reset the buffer for a new batch."""
        self.current_batch_size = 0
        self.current_seq_len = 0

    def write(
        self,
        data: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
    ) -> None:
        """Write data to the buffer.

        Args:
            data: Data to write [seq_len] or scalar.
            batch_idx: Batch index.
            seq_idx: Sequence position.
        """
        if data.dim() == 0:
            # Scalar
            self.buffer[batch_idx, seq_idx] = data
        else:
            # Sequence
            seq_len = data.shape[0]
            self.buffer[batch_idx, seq_idx : seq_idx + seq_len] = data

        self.current_batch_size = max(self.current_batch_size, batch_idx + 1)
        self.current_seq_len = max(self.current_seq_len, seq_idx + data.numel())

    def read(self) -> torch.Tensor:
        """Read the current buffer contents.

        Returns:
            Buffer slice with current data.
        """
        return self.buffer[: self.current_batch_size, : self.current_seq_len]

    def get_async(self) -> MetalAsyncOutput:
        """Get buffer contents as async output.

        Returns:
            MetalAsyncOutput wrapping the buffer.
        """
        return MetalAsyncOutput(tensor=self.read().clone())


def metal_async_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> MetalAsyncOutput:
    """Perform async tensor copy on Metal.

    On Apple Silicon with unified memory, this is effectively
    a synchronous operation since CPU and GPU share memory.

    Args:
        src: Source tensor.
        dst: Destination tensor.

    Returns:
        MetalAsyncOutput for the copy operation.
    """
    # Direct copy (no async needed with unified memory)
    dst.copy_(src)
    return MetalAsyncOutput(tensor=dst)
