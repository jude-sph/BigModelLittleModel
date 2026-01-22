"""Base model interface for MLX models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


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
    """Abstract base class for MLX model wrappers."""

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
