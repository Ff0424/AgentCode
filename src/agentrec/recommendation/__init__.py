"""Stable contracts and artifact loading for recommendation serving."""

from .artifacts import CanonicalUserResolver, ServingArtifacts, load_serving_artifacts
from .catalog import ProductCatalogIndex, RuntimeProduct
from .contracts import RecommendationRequest, RecommendationResult, RecommendedProduct
from .service import RecommendationService

__all__ = [
    "RecommendationRequest",
    "RecommendationService",
    "RecommendationResult",
    "RecommendedProduct",
    "ProductCatalogIndex",
    "RuntimeProduct",
    "CanonicalUserResolver",
    "ServingArtifacts",
    "load_serving_artifacts",
]
