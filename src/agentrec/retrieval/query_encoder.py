"""Provider-neutral query encoding with a local BGE-M3 implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


class QueryEncoder(Protocol):
    @property
    def embedding_dimension(self) -> int: ...

    def encode(self, queries: Sequence[str]) -> np.ndarray: ...


def normalize_query_embeddings(vectors: object, rows: int, dimension: int) -> np.ndarray:
    """Convert provider output to validated contiguous normalized float32 rows."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, dimension):
        raise ValueError(
            f"Query encoder returned shape {array.shape}; expected ({rows}, {dimension})."
        )
    if not np.isfinite(array).all():
        raise ValueError("Query encoder returned NaN or Inf values.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Query encoder returned a non-finite or zero vector.")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


class BGEM3QueryEncoder:
    """Encode queries with the same dense-vector contract as V2-09.2a."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "cuda:0",
        use_fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 2048,
        embedding_dimension: int = 1024,
        model: Any | None = None,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size), ("max_length", max_length),
            ("embedding_dimension", embedding_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string.")
        self._dimension = embedding_dimension
        self._batch_size = batch_size
        self._max_length = max_length

        if model is not None:
            self._model = model
            return
        if model_path is None:
            raise ValueError("model_path is required when model is not injected.")
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local BGE-M3 model directory not found: {path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - GPU server path
            raise RuntimeError("BGE-M3 query encoding requires torch and FlagEmbedding.") from exc
        if device != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("The frozen query encoder contract requires available cuda:0.")
        torch.cuda.set_device(0)
        self._model = BGEM3FlagModel(str(path), use_fp16=use_fp16, devices=[device])

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _validate_queries(queries: Sequence[str]) -> tuple[str, ...]:
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise TypeError("queries must be a non-string sequence of strings.")
        if not queries:
            raise ValueError("queries must not be empty.")
        normalized: list[str] = []
        for index, query in enumerate(queries):
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"queries[{index}] must be a non-empty string.")
            normalized.append(query.strip())
        return tuple(normalized)

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        normalized = self._validate_queries(queries)
        result = self._model.encode(
            list(normalized), batch_size=self._batch_size,
            max_length=self._max_length, return_dense=True,
            return_sparse=False, return_colbert_vecs=False,
        )
        if not isinstance(result, dict) or "dense_vecs" not in result:
            raise ValueError("BGE-M3 output does not contain dense_vecs.")
        return normalize_query_embeddings(
            result["dense_vecs"], len(normalized), self._dimension
        )
