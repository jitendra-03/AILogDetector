"""
Embedding Generator
===================
Generates vector embeddings for log lines. Uses transformers when available,
falls back to lightweight hash-based embeddings for Python 3.14 compatibility.
"""

import hashlib
import logging
import struct
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Maximum number of characters passed to the encoder
_MAX_CHARS = 512
# Dimension of embeddings
_EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """
    Generates text embeddings using transformers (with hash-based fallback).
    
    When torch is unavailable (e.g., Python 3.14), uses a lightweight
    hash-based deterministic embedding instead.

    Config keys consumed (under ``embedding``):
        model:   HuggingFace model name  (default: all-MiniLM-L6-v2)
        device:  "cpu" or "cuda"         (default: cpu)
    """

    def __init__(self, config: dict) -> None:
        emb = config.get("embedding", {})
        self._model_name: str = emb.get("model", "all-MiniLM-L6-v2")
        self._device:     str = emb.get("device", "cpu")
        self._model = None
        self._tokenizer = None
        self._use_torch = self._check_torch_available()
        
        if not self._use_torch:
            logger.info("Using hash-based embeddings (torch unavailable)")

    def _check_torch_available(self) -> bool:
        """Check if torch can be imported without errors."""
        try:
            import torch
            torch.tensor([1.0])
            return True
        except Exception as e:
            logger.debug("Torch not available: %s", e)
            return False

    @property
    def model(self):
        """Lazy-load the transformer model."""
        if self._model is None and self._use_torch:
            try:
                from transformers import AutoTokenizer, AutoModel
                logger.info("Loading embedding model: %s", self._model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModel.from_pretrained(self._model_name)
                logger.info("Embedding model ready (%s)", self._model_name)
            except Exception as e:
                logger.warning("Failed to load torch model, using hash fallback: %s", e)
                self._use_torch = False
        return self._model

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Encode one or more strings into fixed-dimension embeddings.

        Returns:
            List[float]       for a single string input.
            List[List[float]] for a list input.
        """
        if isinstance(text, str):
            return self._encode_single(text)
        return [self._encode_single(t) for t in text]

    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text string."""
        truncated = text[:_MAX_CHARS]
        
        if self._use_torch and self.model is not None:
            try:
                import torch
                inputs = self._tokenizer(truncated, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                vec = embeddings[0].numpy().tolist()
                return vec[:_EMBEDDING_DIM] + [0.0] * max(0, _EMBEDDING_DIM - len(vec))
            except Exception as e:
                logger.debug("Torch embedding failed, using fallback: %s", e)
        
        return self._hash_embedding(truncated)

    @staticmethod
    def _hash_embedding(text: str) -> List[float]:
        """Generate a deterministic embedding from text using cryptographic hash."""
        h = hashlib.sha256(text.encode('utf-8')).digest()
        embeddings = []
        for i in range(0, len(h), 4):
            chunk = h[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            val = struct.unpack('>f', chunk)[0]
            embeddings.append(val / (2**31))
        
        while len(embeddings) < _EMBEDDING_DIM:
            embeddings.extend(embeddings[:min(len(embeddings), _EMBEDDING_DIM - len(embeddings))])
        
        return embeddings[:_EMBEDDING_DIM]

    def encode_error(
        self,
        message: str,
        context: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Encode an error message with optional context lines.
        """
        if context:
            ctx_text = " | ".join(context[-3:])
            combined = f"{message} [context: {ctx_text}]"
        else:
            combined = message
        return self.encode(combined)
