# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx

class MetalRejectionSampler:
    """Performs MLX-native rejection sampling for speculative decoding."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def sample(self, target_logits: mx.array, draft_logits: mx.array, draft_tokens: mx.array) -> int:
        """Compares target and draft probabilities to find first rejection.
        
        Args:
            target_logits: Shape (k, vocab_size)
            draft_logits: Shape (k, vocab_size)
            draft_tokens: Shape (k,)
            
        Returns:
            int: Number of accepted tokens (m).
        """
        # Placeholder: accept all for now to show the flow
        return draft_tokens.shape[0]
