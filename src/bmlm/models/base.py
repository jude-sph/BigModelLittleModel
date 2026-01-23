"""Base model interface for MLX models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from PIL import Image


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class GenerationResult:
    """Result from model generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


class BaseModel(ABC):
    """Abstract base class for MLX text-only model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return
        self.model, self.tokenizer = load(self.config.model_path)
        self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        mx.clear_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _generate(self, prompt: str) -> GenerationResult:
        """Run generation and return result with timing."""
        import time

        if not self._loaded:
            self.load()

        # Create sampler with temperature and top_p
        sampler = make_sampler(temp=self.config.temperature, top_p=self.config.top_p)

        start = time.perf_counter()
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            sampler=sampler,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Estimate tokens (rough approximation)
        tokens = len(self.tokenizer.encode(response))

        return GenerationResult(
            text=response,
            tokens_generated=tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        )

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Model-specific generation method."""
        pass


class VisionModel(ABC):
    """Abstract base class for MLX vision-language model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.model_config = None
        self._loaded = False

    def load(self) -> None:
        """Load the vision model and processor."""
        if self._loaded:
            return
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config

        self.model, self.processor = vlm_load(self.config.model_path)
        self.model_config = load_config(self.config.model_path)

        # Replace fast image processor with slow version to avoid PyTorch tensor issues
        # The fast processor only supports PyTorch tensors, which breaks mlx_vlm
        try:
            from transformers import AutoImageProcessor
            slow_processor = AutoImageProcessor.from_pretrained(
                self.config.model_path, use_fast=False
            )
            self.processor.image_processor = slow_processor
        except Exception:
            pass  # If it fails, hope the default works

        self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        self.model = None
        self.processor = None
        self.model_config = None
        self._loaded = False
        mx.clear_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _generate_with_image(self, prompt: str, image: Image.Image) -> GenerationResult:
        """Run generation with an image and return result with timing."""
        import tempfile
        import time

        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if not self._loaded:
            self.load()

        # mlx_vlm expects image paths, not PIL Images - save to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        # Apply chat template for the vision model
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model_config,
            prompt,
            num_images=1,
        )

        start = time.perf_counter()
        response = vlm_generate(
            self.model,
            self.processor,
            formatted_prompt,
            image_path,  # Pass path string, not list
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Clean up temp file
        import os
        try:
            os.unlink(image_path)
        except:
            pass

        # Estimate tokens (rough approximation based on response length)
        tokens = len(response.split()) * 1.3  # Rough estimate

        return GenerationResult(
            text=response,
            tokens_generated=int(tokens),
            generation_time_ms=elapsed_ms,
            tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        )

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Model-specific generation method."""
        pass
