"""
AgentTools exposes stable, structured tool functions over the existing
recommendation, repository, and RAG context services.

It does not decide when tools should be called and does not call an LLM.
"""

import argparse
import importlib.util
import json
import math
import numbers
import sys
from pathlib import Path


# ============================================================
# 1. Existing local service modules
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PRODUCT_REPOSITORY_PATH = SCRIPTS_DIR / "18_product_repository.py"
RECOMMENDATION_SERVICE_V2_PATH = (
    SCRIPTS_DIR / "19_recommendation_service_v2.py"
)
RAG_CONTEXT_BUILDER_PATH = SCRIPTS_DIR / "20_rag_context_builder.py"


def _load_local_class(module_path: Path, module_name: str, class_name: str):
    """Load one class from a numeric-prefixed script without modifying sys.path."""

    if not module_path.is_file():
        raise FileNotFoundError(f"Required local module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_path}.")

    # Local dynamic imports should not create bytecode cache artifacts.
    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load local module {module_path}: {exc}") from exc

    loaded_class = getattr(module, class_name, None)
    if not isinstance(loaded_class, type):
        raise ImportError(f"{class_name} was not found in {module_path}.")
    return loaded_class


# ============================================================
# 2. Provider-neutral tool schemas
# ============================================================

TOOL_SCHEMAS = [
    {
        "name": "recommend_products",
        "description": (
            "Search and rank products that match a user's shopping request. "
            "Use this when the user asks for product recommendations or "
            "specifies preferences such as product type, features, price, or rating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic shopping query.",
                },
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "candidate_k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                },
                "rerank_pool_k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20,
                },
                "max_price": {"type": ["number", "null"]},
                "min_rating": {"type": ["number", "null"]},
                "min_rating_count": {"type": ["integer", "null"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_product_details",
        "description": "Fetch structured product facts for known product IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["product_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_products",
        "description": (
            "Fetch aligned structured facts for comparing two to five known products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 5,
                    "uniqueItems": True,
                }
            },
            "required": ["product_ids"],
            "additionalProperties": False,
        },
    },
]


# ============================================================
# 3. JSON-safe result boundary
# ============================================================

def _to_json_safe(value):
    """Recursively convert supported values to JSON-serializable built-ins."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Tool results may contain only string dictionary keys; "
                    f"found {type(key).__name__}."
                )
            converted[key] = _to_json_safe(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    raise TypeError(
        "Tool result contains unsupported type "
        f"{type(value).__name__}; implicit string conversion is not allowed."
    )


# ============================================================
# 4. Stable Agent tool layer
# ============================================================

class AgentTools:
    """Expose recommendation and product facts through explicit tool methods."""

    def __init__(
        self,
        recommendation_service=None,
        repository=None,
        context_builder=None,
    ):
        if recommendation_service is None:
            service_class = _load_local_class(
                RECOMMENDATION_SERVICE_V2_PATH,
                "agentcode_recommendation_service_v2_for_tools",
                "RecommendationServiceV2",
            )
            recommendation_service = service_class()
        if not callable(getattr(recommendation_service, "recommend", None)):
            raise TypeError(
                "recommendation_service must provide a callable recommend() method."
            )

        if repository is None:
            shared_repository = getattr(recommendation_service, "repository", None)
            if callable(getattr(shared_repository, "get_many", None)):
                repository = shared_repository
            else:
                repository_class = _load_local_class(
                    PRODUCT_REPOSITORY_PATH,
                    "agentcode_product_repository_18_for_tools",
                    "ProductRepository",
                )
                repository = repository_class()
        if not callable(getattr(repository, "get_many", None)):
            raise TypeError("repository must provide a callable get_many() method.")

        if context_builder is None:
            builder_class = _load_local_class(
                RAG_CONTEXT_BUILDER_PATH,
                "agentcode_rag_context_builder_20_for_tools",
                "RAGContextBuilder",
            )
            context_builder = builder_class()
        if not callable(getattr(context_builder, "build", None)):
            raise TypeError("context_builder must provide a callable build() method.")

        self.recommendation_service = recommendation_service
        self.repository = repository
        self.context_builder = context_builder

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, found {type(query).__name__}.")
        if not query.strip():
            raise ValueError("query must be a non-empty string.")

    @staticmethod
    def _validate_product_ids(product_ids, allow_empty: bool = True) -> None:
        if not isinstance(product_ids, list):
            raise TypeError("product_ids must be a list of strings.")
        if not allow_empty and not product_ids:
            raise ValueError("product_ids must not be empty.")
        for position, product_id in enumerate(product_ids):
            if not isinstance(product_id, str):
                raise TypeError(
                    f"product_ids[{position}] must be a string, found "
                    f"{type(product_id).__name__}."
                )
            if not product_id.strip():
                raise ValueError(
                    f"product_ids[{position}] must be a non-empty string."
                )

    def recommend_products(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        rerank_pool_k: int = 20,
        max_price: float | None = None,
        min_rating: float | None = None,
        min_rating_count: int | None = None,
    ) -> dict:
        """Return V2 recommendations plus their deterministic RAG context."""

        self._validate_query(query)
        recommendations = self.recommendation_service.recommend(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            rerank_pool_k=rerank_pool_k,
            max_price=max_price,
            min_rating=min_rating,
            min_rating_count=min_rating_count,
        )
        context_result = self.context_builder.build(query, recommendations)
        result = {
            "tool": "recommend_products",
            "query": context_result["query"],
            "recommendations": recommendations,
            "rag_context": context_result["context"],
            "included_product_ids": context_result["included_product_ids"],
            "recommendation_count": int(len(recommendations)),
            "context_chars": int(context_result["context_chars"]),
            "context_truncated": bool(context_result["truncated"]),
        }
        return _to_json_safe(result)

    def get_product_details(self, product_ids: list[str]) -> dict:
        """Fetch products directly, preserving requested order and duplicates."""

        self._validate_product_ids(product_ids)
        products = self.repository.get_many(product_ids)
        if not isinstance(products, list) or len(products) != len(product_ids):
            raise RuntimeError(
                "Repository result count does not match requested product IDs."
            )
        result = {
            "tool": "get_product_details",
            "requested_product_ids": list(product_ids),
            "products": products,
            "count": int(len(products)),
        }
        return _to_json_safe(result)

    def compare_products(self, product_ids: list[str]) -> dict:
        """Return aligned facts without drawing comparative conclusions."""

        self._validate_product_ids(product_ids, allow_empty=False)
        if not 2 <= len(product_ids) <= 5:
            raise ValueError("compare_products requires between 2 and 5 product IDs.")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("compare_products does not allow duplicate product IDs.")

        products = self.repository.get_many(product_ids)
        if not isinstance(products, list) or len(products) != len(product_ids):
            raise RuntimeError(
                "Repository result count does not match comparison product IDs."
            )
        for position, (product_id, product) in enumerate(zip(product_ids, products)):
            if not isinstance(product, dict):
                raise RuntimeError(
                    f"Repository product at position {position} is not a dictionary."
                )
            if product.get("product_id") != product_id:
                raise RuntimeError(
                    "Repository results are not aligned with requested IDs at "
                    f"position {position}: {product_id!r} != "
                    f"{product.get('product_id')!r}."
                )

        comparison_fields = (
            "title",
            "price",
            "average_rating",
            "rating_number",
            "store",
            "categories",
            "features",
            "product_details",
            "description",
        )
        comparison_products = [
            {
                "product_id": product_id,
                **{field: product.get(field) for field in comparison_fields},
            }
            for product_id, product in zip(product_ids, products)
        ]
        comparison = {
            field: [
                {"product_id": product_id, "value": product.get(field)}
                for product_id, product in zip(product_ids, products)
            ]
            for field in comparison_fields
        }
        result = {
            "tool": "compare_products",
            "product_ids": list(product_ids),
            "products": comparison_products,
            "comparison": comparison,
            "count": int(len(products)),
        }
        return _to_json_safe(result)

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch only explicitly approved public tools."""

        if not isinstance(tool_name, str):
            raise TypeError("tool_name must be a string.")
        if not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string.")
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be a dictionary.")

        dispatch = {
            "recommend_products": self.recommend_products,
            "get_product_details": self.get_product_details,
            "compare_products": self.compare_products,
        }
        if tool_name not in dispatch:
            raise ValueError(f"Unknown tool: {tool_name!r}.")
        return dispatch[tool_name](**arguments)


# ============================================================
# 5. Minimal JSON stdout CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run one local AgentRec tool.")
    parser.add_argument(
        "--tool",
        required=True,
        choices=("recommend", "details", "compare"),
    )
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-pool-k", type=int, default=20)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument("--min-rating-count", type=int, default=None)
    parser.add_argument("--product-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools = AgentTools()

    if args.tool == "recommend":
        if args.query is None:
            raise ValueError("recommend mode requires --query.")
        result = tools.recommend_products(
            query=args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank_pool_k=args.rerank_pool_k,
            max_price=args.max_price,
            min_rating=args.min_rating,
            min_rating_count=args.min_rating_count,
        )
    elif args.tool == "details":
        if not args.product_id:
            raise ValueError("details mode requires at least one --product-id.")
        result = tools.get_product_details(args.product_id)
    else:
        if not 2 <= len(args.product_id) <= 5:
            raise ValueError("compare mode requires 2 to 5 --product-id values.")
        result = tools.compare_products(args.product_id)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
