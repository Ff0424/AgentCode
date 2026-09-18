"""Deterministic query-encoder tests without CUDA or a real BGE-M3 model."""

from __future__ import annotations

import unittest

import numpy as np

from src.agentrec.retrieval import BGEM3QueryEncoder


class FakeModel:
    def __init__(self, values: object) -> None:
        self.values = values
        self.calls: list[dict] = []

    def encode(self, queries, **kwargs):
        self.calls.append({"queries": queries, **kwargs})
        return {"dense_vecs": self.values}


class ChunkQueryEncoderTests(unittest.TestCase):
    def test_output_is_float32_contiguous_and_normalized(self) -> None:
        model = FakeModel(np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64))
        encoder = BGEM3QueryEncoder(model=model, embedding_dimension=2)
        result = encoder.encode([" first ", "second"])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
        self.assertEqual(model.calls[0]["queries"], ["first", "second"])
        self.assertFalse(model.calls[0]["return_sparse"])

    def test_query_validation(self) -> None:
        encoder = BGEM3QueryEncoder(model=FakeModel([[1.0, 0.0]]), embedding_dimension=2)
        for invalid in ([], "query", [""], ["   "], [123]):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                encoder.encode(invalid)

    def test_shape_finite_and_zero_vector_refusal(self) -> None:
        cases = (
            (np.ones((1, 3)), "shape"),
            (np.asarray([[np.nan, 1.0]]), "NaN or Inf"),
            (np.zeros((1, 2)), "zero vector"),
        )
        for values, message in cases:
            with self.subTest(message=message):
                encoder = BGEM3QueryEncoder(model=FakeModel(values), embedding_dimension=2)
                with self.assertRaisesRegex(ValueError, message):
                    encoder.encode(["query"])


if __name__ == "__main__":
    unittest.main()
