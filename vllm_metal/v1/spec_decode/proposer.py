# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Any
from typing import List, Optional
import torch
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from vllm.config import VllmConfig
from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_metal.config import get_config
from vllm_metal.v1.model_lifecycle import ModelLifecycle
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.utils import get_model_download_path

logger = init_logger(__name__)


class MinimalDraftRunner:
    """
    A lightweight mock runner to satisfy the ModelLifecycle interface.
    Ensures the draft model only loads weights without allocating paged resources.
    """

    def __init__(self, vllm_config: VllmConfig, draft_model_path: str):
        self.vllm_config = vllm_config
        self.metal_config = get_config()
        self.device = torch.device("cpu")

        # Clone model config but point it to the draft model
        self.model_config = copy.copy(vllm_config.model_config)
        self.model_config.model = draft_model_path

        # Required stubs for ModelLifecycle
        self.model_args: dict[str, Any] = {}
        self.model: Any = None
        self.tokenizer: Any = None
        self._vocab_size: int = 0
        self._is_vlm = False
        self._is_stt = False

        self._model_adapter = DefaultModelAdapter()
        self._model_lifecycle = ModelLifecycle(self, self._model_adapter)

    @property
    def is_mla(self) -> bool:
        """MLA models (GLM/DeepSeek) have no standard q/k/v proj."""
        return "kv_lora_rank" in self.model_args

    @property
    def is_hybrid(self) -> bool:
        """Hybrid models (Qwen) mix SDPA and linear attention."""
        fai = self.model_args.get("full_attention_interval", 0)
        return isinstance(fai, int) and fai > 0

    def load_model(self):
        logger.info("Calling ModelLifecycle.load() for %s...", self.model_config.model)
        self._model_lifecycle.load()
        logger.info("ModelLifecycle.load() completed.")


class MetalDraftProposer:
    """
    MLX-native Speculative Proposer

    Loads a small model (e.g. Qwen2.5-0.5B) and generate k speculative tokens
    using a lazy autoregressive loop
    """

    def __init__(self, vllm_config: VllmConfig, model_path: Optional[str] = None):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.model_config = vllm_config.model_config

        # draft model
        self.model_path = model_path or get_model_download_path(
            self.speculative_config.model
        )
        self._draft_runner = MinimalDraftRunner(vllm_config, self.model_path)

    def load_model(self) -> None:
        logger.info(f"Loading draft model from: {self.model_path}")
        self._draft_runner.load_model()

    @property
    def model(self):
        return self._draft_runner.model

    def propose(self, input_ids: mx.array, k: int) -> tuple[mx.array, mx.array]:
        """
        Generate k speculative tokens as a lazy computation graph.
        """

        # 1. initialize draft cache
        # use a standard contiguous cache (no paged blocks) to minimize overhead.
        logger.info("Initializing draft cache for model of type %s", type(self.model))
        cache = make_prompt_cache(self.model)

        # 2. initial pre-fill pre-loop
        # run the current request history through the draft model to populate the cache.
        logger.info("Running prefill for %d tokens...", input_ids.shape[1])
        logits = self.model(input_ids, cache=cache)
        logits = mx.stop_gradient(logits)

        draft_tokens = []
        draft_logits = []

        # step 3: lazy autoregressive loop

        # grab the very first predicted token from the prefill step
        current_token = mx.argmax(logits[:, -1, :], axis=-1)

        # build the computation graph
        logger.info("Building lazy graph for %d steps...", k)
        for i in range(k):
            draft_tokens.append(current_token)

            # format token for next forward pass
            token_input = (
                current_token[:, None] if current_token.ndim == 1 else current_token
            )
            step_logits = self.model(token_input, cache=cache)
            step_logits = mx.stop_gradient(step_logits)

            draft_logits.append(step_logits)

            # Predict the next token to feed into the next iteration
            current_token = mx.argmax(step_logits[:, -1, :], axis=-1)
            logger.debug("  - Step %d/%d added to graph", i + 1, k)

        # Return the un-evaluated arrays. The GPU won't actually do the math
        # for Step 3 until the SpecWorker orchestrator explicitly asks for it.
        logger.info("Graph building complete. Returning to caller.")
        return (
            mx.stack(draft_tokens, axis=1),
            mx.concatenate(draft_logits, axis=1),
        )
