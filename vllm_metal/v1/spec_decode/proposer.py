# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import mlx.core as mx

class MetalDraftProposer:
    """Proposes draft tokens using a small model on Metal."""
    
    def __init__(self, vllm_config, draft_config):
        self.vllm_config = vllm_config
        self.draft_config = draft_config
        # TODO: Initialize draft model and lifecycle here
        self.k = getattr(draft_config, "num_speculative_tokens", 5)

    def propose(self, request_ids: List[str]) -> mx.array:
        """Generate k draft tokens for each request.
        
        Returns:
            mx.array: Shape (batch_size, k) of draft token IDs.
        """
        # Placeholder: returning zeros for now to show the flow
        batch_size = len(request_ids)
        return mx.zeros((batch_size, self.k), dtype=mx.int32)
