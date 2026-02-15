"""
Sentence-BERT encoder for language-conditioned policy.

Wraps the sentence-transformers library with caching to avoid
redundant encoding of the same descriptor strings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict


class LanguageEncoder:
    """Encodes natural language task descriptors into fixed-size embeddings.

    Uses Sentence-BERT (all-MiniLM-L6-v2 by default) for fast, lightweight
    encoding. Embeddings are cached to avoid recomputation.

    Args:
        model_name: HuggingFace model name for sentence-transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Lazy load
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimensionality."""
        # all-MiniLM-L6-v2 produces 384-dim embeddings
        return self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode a text descriptor into a fixed-size embedding vector.

        Args:
            text: Natural language descriptor (e.g., "lift gently").

        Returns:
            Normalized embedding vector, shape (embedding_dim,).
        """
        if text not in self._cache:
            embedding = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            self._cache[text] = embedding.astype(np.float32)
        return self._cache[text]

    def encode_batch(self, texts: list) -> np.ndarray:
        """Encode multiple descriptors.

        Args:
            texts: List of descriptor strings.

        Returns:
            Embeddings array, shape (len(texts), embedding_dim).
        """
        # Check cache first
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = self.model.encode(
                uncached, convert_to_numpy=True, normalize_embeddings=True
            )
            for text, emb in zip(uncached, embeddings):
                self._cache[text] = emb.astype(np.float32)

        return np.array([self._cache[t] for t in texts])

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
