# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible block table operations.

This module provides block table management using PyTorch/MLX operations
instead of Triton kernels.
"""

import torch


class MetalBlockTables:
    """Block table manager for Metal backend.

    This class manages the mapping between logical blocks (per-sequence)
    and physical blocks (in the KV cache). It replaces the Triton-based
    implementation with pure PyTorch operations.
    """

    def __init__(
        self,
        max_num_blocks_per_seq: int,
        max_num_seqs: int,
        device: torch.device,
    ):
        """Initialize block tables.

        Args:
            max_num_blocks_per_seq: Maximum blocks per sequence.
            max_num_seqs: Maximum number of sequences.
            device: Torch device for tensors.
        """
        self.max_num_blocks_per_seq = max_num_blocks_per_seq
        self.max_num_seqs = max_num_seqs
        self.device = device

        # Initialize block table tensor
        # Shape: [max_num_seqs, max_num_blocks_per_seq]
        self.block_table = torch.zeros(
            (max_num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        # Track number of blocks per sequence
        self.num_blocks_per_seq = torch.zeros(
            max_num_seqs,
            dtype=torch.int32,
            device=device,
        )

    def reset(self) -> None:
        """Reset all block tables."""
        self.block_table.zero_()
        self.num_blocks_per_seq.zero_()

    def get_block_table(self, seq_idx: int) -> torch.Tensor:
        """Get block table for a specific sequence.

        Args:
            seq_idx: Sequence index.

        Returns:
            Block table tensor for the sequence.
        """
        num_blocks = int(self.num_blocks_per_seq[seq_idx])
        return self.block_table[seq_idx, :num_blocks]

    def set_block_table(
        self,
        seq_idx: int,
        block_numbers: list[int] | torch.Tensor,
    ) -> None:
        """Set block table for a specific sequence.

        Args:
            seq_idx: Sequence index.
            block_numbers: List or tensor of block numbers.
        """
        if isinstance(block_numbers, list):
            block_numbers = torch.tensor(
                block_numbers,
                dtype=torch.int32,
                device=self.device,
            )

        num_blocks = len(block_numbers)
        self.block_table[seq_idx, :num_blocks] = block_numbers
        self.num_blocks_per_seq[seq_idx] = num_blocks

    def append_block(self, seq_idx: int, block_number: int) -> None:
        """Append a block to a sequence's block table.

        Args:
            seq_idx: Sequence index.
            block_number: Block number to append.
        """
        num_blocks = int(self.num_blocks_per_seq[seq_idx])
        if num_blocks >= self.max_num_blocks_per_seq:
            raise RuntimeError(
                f"Cannot append block: sequence {seq_idx} has reached "
                f"max blocks ({self.max_num_blocks_per_seq})"
            )

        self.block_table[seq_idx, num_blocks] = block_number
        self.num_blocks_per_seq[seq_idx] = num_blocks + 1

    def get_slot_mapping(
        self,
        seq_idx: int,
        start_pos: int,
        num_tokens: int,
        block_size: int,
    ) -> torch.Tensor:
        """Compute slot mapping for tokens in a sequence.

        Args:
            seq_idx: Sequence index.
            start_pos: Starting position in the sequence.
            num_tokens: Number of tokens.
            block_size: Size of each cache block.

        Returns:
            Slot mapping tensor [num_tokens].
        """
        # Compute which block and offset each position maps to
        positions = torch.arange(
            start_pos,
            start_pos + num_tokens,
            device=self.device,
        )
        block_indices = positions // block_size
        block_offsets = positions % block_size

        # Get physical block numbers
        block_table = self.block_table[seq_idx]
        physical_blocks = block_table[block_indices.long()]

        # Compute slot indices
        slot_mapping = physical_blocks * block_size + block_offsets

        return slot_mapping.int()

    def get_batch_block_table(
        self,
        seq_indices: torch.Tensor | list[int],
    ) -> torch.Tensor:
        """Get block tables for a batch of sequences.

        Args:
            seq_indices: Indices of sequences in the batch.

        Returns:
            Block table tensor [batch_size, max_num_blocks_per_seq].
        """
        if isinstance(seq_indices, list):
            seq_indices = torch.tensor(
                seq_indices,
                dtype=torch.long,
                device=self.device,
            )

        return self.block_table[seq_indices]

    def copy_block_table(self, src_seq_idx: int, dst_seq_idx: int) -> None:
        """Copy block table from source to destination sequence.

        Args:
            src_seq_idx: Source sequence index.
            dst_seq_idx: Destination sequence index.
        """
        num_blocks = int(self.num_blocks_per_seq[src_seq_idx])
        self.block_table[dst_seq_idx, :num_blocks] = self.block_table[
            src_seq_idx, :num_blocks
        ]
        self.num_blocks_per_seq[dst_seq_idx] = num_blocks

    def fork(
        self,
        src_seq_idx: int,
        dst_seq_indices: list[int],
    ) -> None:
        """Fork a sequence's block table to multiple destinations.

        Used in beam search to duplicate cache for beam candidates.

        Args:
            src_seq_idx: Source sequence index.
            dst_seq_indices: List of destination sequence indices.
        """
        for dst_idx in dst_seq_indices:
            self.copy_block_table(src_seq_idx, dst_idx)
