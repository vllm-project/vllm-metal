# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for vLLM Metal."""

import argparse
import logging
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vllm_metal.config import get_config

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="vLLM Metal API",
    description="OpenAI-compatible API for LLM inference on Apple Silicon",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_engine: Any = None
_model_name: str = ""


# Request/Response models
class ImageUrl(BaseModel):
    """Image URL for vision models."""

    url: str


class ContentPart(BaseModel):
    """Content part for multimodal messages."""

    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    """Chat message with optional multimodal content."""

    role: str
    content: str | list[ContentPart]  # String or array of content parts


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_completion_tokens: int | None = None
    max_tokens: int | None = None  # Deprecated, use max_completion_tokens
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | str | None = None

    @property
    def effective_max_tokens(self) -> int:
        """Get effective max_tokens, preferring max_completion_tokens."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return 16  # Default per vLLM


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = 16  # Default per vLLM
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | str | None = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm-metal"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# Health check
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    if not _model_name:
        return ModelsResponse(data=[])

    return ModelsResponse(
        data=[
            ModelInfo(
                id=_model_name,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle chat completion requests, including vision/multimodal."""
    global _engine

    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format messages into a prompt and extract images
    prompt, images = format_chat_prompt(request.messages)

    # Generate response
    try:
        output = _engine.generate(
            prompt=prompt,
            max_tokens=request.effective_max_tokens,
            temperature=request.temperature,
            images=images if images else None,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Count tokens (approximate)
    prompt_tokens = len(prompt.split())
    completion_tokens = len(output.split())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    """Handle completion requests."""
    global _engine

    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle single or multiple prompts
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    max_tokens = request.max_tokens if request.max_tokens is not None else 16

    for i, prompt in enumerate(prompts):
        try:
            output = _engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        prompt_tokens = len(prompt.split())
        completion_tokens = len(output.split())
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        choices.append(
            CompletionChoice(
                index=i,
                text=output,
                finish_reason="stop",
            )
        )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


def extract_message_content(content: str | list[ContentPart]) -> tuple[str, list[str]]:
    """Extract text and images from message content.

    Args:
        content: String content or list of content parts

    Returns:
        Tuple of (text_content, list_of_image_urls)
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    images = []

    for part in content:
        if part.type == "text" and part.text:
            text_parts.append(part.text)
        elif part.type == "image_url" and part.image_url:
            images.append(part.image_url.url)

    return " ".join(text_parts), images


def format_chat_prompt(messages: list[Message]) -> tuple[str, list[str]]:
    """Format chat messages into a prompt string and extract images.

    Uses a simple format that works with most chat models.

    Returns:
        Tuple of (formatted_prompt, list_of_image_urls)
    """
    formatted = []
    all_images: list[str] = []

    for msg in messages:
        text, images = extract_message_content(msg.content)
        all_images.extend(images)

        if msg.role == "system":
            formatted.append(f"System: {text}")
        elif msg.role == "user":
            formatted.append(f"User: {text}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {text}")
        else:
            formatted.append(f"{msg.role}: {text}")

    formatted.append("Assistant:")
    return "\n".join(formatted), all_images


def create_engine(model_name: str) -> Any:
    """Create the inference engine.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Model runner instance
    """
    from vllm_metal.model_runner import MetalModelRunner

    # Create a minimal config
    class MinimalModelConfig:
        def __init__(self, model: str):
            self.model = model

    class MinimalVllmConfig:
        def __init__(self, model: str):
            self.model_config = MinimalModelConfig(model)

    config = MinimalVllmConfig(model_name)
    runner = MetalModelRunner(config)  # type: ignore[arg-type]
    runner.load_model()

    return runner


def main() -> None:
    """Main entry point for the server."""
    global _engine, _model_name

    parser = argparse.ArgumentParser(description="vLLM Metal API Server")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = get_config()
    if config.debug:
        logger.info(f"Metal config: {config}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    _model_name = args.model
    _engine = create_engine(args.model)
    logger.info("Model loaded successfully")

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
