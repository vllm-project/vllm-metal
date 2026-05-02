# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
from typing import Tuple


class MetalRejectionSampler:
    """Performs MLX-native rejection sampling for speculative decoding."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def sample(
        self,
        target_logits: mx.array,  # (k+1, vocab) from scorer
        draft_logits: mx.array,  # (k, vocab) from proposer
        draft_tokens: mx.array,  # (k,) tokens guessed by Drafter
    ) -> Tuple[int, int]:
        """
        Compares target and draft probabilities to find first rejection.

        Args:
            `target_logits`: Shape (k, vocab_size)
            `draft_logits`: Shape (k, vocab_size)
            `draft_tokens`: Shape (k,)

        Returns:
            - `num_accepted`: how many draft tokens passed the check
            - `bonus_token`: the final token to append to the sequence
        """

        # 1. Convert to probabilities
        # we ignore the last token(scorer's prediction to next token that hasn't been guessed yet)
        p_target = mx.softmax(target_logits[:-1, :], axis=-1)  # (k, vocab)
        q_draft = mx.softmax(draft_logits, axis=-1)  # (k, vocab)

        # 2. Extract probabilities for draft model's specific choices
        indices = mx.arange(draft_tokens.shape[0])
        p_target_at_draft = p_target[indices, draft_tokens]
        q_draft_at_draft = q_draft[indices, draft_tokens]

        # 3. Calculate the acceptance probability
        # how much target agrees with the draft model
        acceptance_prob = mx.minimum(1.0, p_target_at_draft / q_draft_at_draft)

        # 4. Parallel rejection check
        random_thresholds = mx.random.uniform(shape=acceptance_prob.shape)
        is_accepted_mask = random_thresholds < acceptance_prob

        # 5. determine the divergent point
        # find the first rejection(false) in the mask
        first_rejection_idx = mx.argmin(is_accepted_mask).item()
        is_entire_chunk_valid = mx.all(is_accepted_mask).item()

        if is_entire_chunk_valid:
            num_accepted = draft_tokens.shape[0]

            # if all are accepted, sample a new 'bonus' token from the final logit
            bonus_token = mx.argmax(target_logits[-1, :], axis=-1).item()
        else:
            num_accepted = int(first_rejection_idx)

            # if a rejection occured, resample from the residual distribution
            p_target_rejected = p_target[num_accepted]
            q_draft_rejected = q_draft[num_accepted]

            residual_distribution = mx.maximum(
                0.0, p_target_rejected - q_draft_rejected
            )
            bonus_token = mx.argmax(residual_distribution, axis=-1).item()

        return num_accepted, int(bonus_token)
