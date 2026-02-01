"""Base model interface for PyTorch/CUDA models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"  # "cuda" or "cpu"
    load_in_4bit: bool = True  # Use 4-bit quantization for memory efficiency


@dataclass
class GenerationResult:
    """Result from model generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


class BaseModel(ABC):
    """Abstract base class for PyTorch text-only model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Configure quantization for memory efficiency
        if self.config.load_in_4bit and self.config.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _generate(self, prompt: str) -> GenerationResult:
        """Run generation and return result with timing."""
        import time

        if not self._loaded:
            self.load()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Decode only the generated tokens (not the input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        tokens = len(generated_tokens)

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
    """Abstract base class for PyTorch vision-language model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load the vision model and processor."""
        if self._loaded:
            return

        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        # Configure quantization for memory efficiency
        if self.config.load_in_4bit and self.config.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        self._loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _generate_with_image(self, prompt: str, image: Image.Image) -> GenerationResult:
        """Run generation with an image and return result with timing."""
        import time

        if not self._loaded:
            self.load()

        # Format the message for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template and process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Decode only the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)

        tokens = len(generated_tokens)

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
