"""
构建适用于 BGE-M3 的 Product Documents V2。

相比 V1 的主要改进：

1. 根据真实数据分析调整字段优先级
2. 使用 BGE-M3 tokenizer 控制真实 Token 长度
3. 将 Product Document 控制在约 2048 tokens 内
4. 优先保留高价值商品属性
5. Description 使用剩余 Token Budget
6. 仍然保持：
   1 Product = 1 Document

输入：
data/processed/recommendation/parent_asins_10core.json
data/raw/meta_Electronics.jsonl.gz

输出：
data/processed/rag/product_documents_v2.jsonl
"""

import gzip
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

META_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "meta_Electronics.jsonl.gz"
)

TEN_CORE_ASIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation"
    / "parent_asins_10core.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "product_documents_v2.jsonl"
)


# ============================================================
# 2. Embedding / Token 配置
# ============================================================

MODEL_NAME = "BAAI/bge-m3"

# 根据 10_analyze_bge_token_lengths.py 的真实统计结果：
#
# > 2048 tokens 的商品仅占 1.33%
#
# 因此 V2 暂时使用 2048 作为 Product Document Token Budget。
MAX_DOCUMENT_TOKENS = 2048


# ============================================================
# 3. Product Details 配置
# ============================================================

EXCLUDED_DETAIL_KEYS = {
    "Date First Available",
    "Best Sellers Rank",
    "Is Discontinued By Manufacturer",
    "Part Number",
    "Manufacturer Part Number",
}

# 单个 detail value 的保护上限。
#
# 这是对异常 Amazon Metadata 的第一层保护。
# 后面还有整个 Document 的 Token Budget 作为第二层保护。
MAX_DETAIL_VALUE_LENGTH = 500


# ============================================================
# 4. 基础文本清洗
# ============================================================

def clean_text(value):
    """
    对 Metadata 文本进行轻量清洗。

    当前处理：
    1. None
    2. 多余空白字符
    3. Amazon 页面残留的 "See more"
    """

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"\s*See more\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_list(values):
    """
    清洗 list 类型字段。

    例如：
    [
        "Bluetooth 5.0",
        "100W Power Delivery"
    ]
    """

    if not isinstance(values, list):
        return []

    cleaned_values = []

    for value in values:

        cleaned = clean_text(value)

        if cleaned:
            cleaned_values.append(cleaned)

    return cleaned_values


# ============================================================
# 5. Details 转换
# ============================================================

def build_details_text(details):
    """
    将 Product Details 转换为可读文本。

    V2 仍然采用小型 denylist，
    而不是维护包含上千字段的 whitelist。
    """

    if not isinstance(details, dict):
        return ""

    detail_parts = []

    for key, value in details.items():

        if key in EXCLUDED_DETAIL_KEYS:
            continue

        cleaned_key = clean_text(key)
        cleaned_value = clean_text(value)

        if not cleaned_key or not cleaned_value:
            continue

        # 防止单个异常属性过长
        if len(cleaned_value) > MAX_DETAIL_VALUE_LENGTH:

            cleaned_value = (
                cleaned_value[:MAX_DETAIL_VALUE_LENGTH]
                + "..."
            )

        detail_parts.append(
            f"{cleaned_key}: {cleaned_value}"
        )

    return " | ".join(detail_parts)


# ============================================================
# 6. 构建各字段 Section
# ============================================================

def build_sections(product):
    """
    将商品拆成具有明确优先级的 Section。

    顺序非常重要：

    Title
        ↓
    Category
        ↓
    Features
        ↓
    Product Details
        ↓
    Description

    Description 放在最后，
    避免超长营销文本挤占高价值属性。
    """

    sections = []

    # --------------------------------------------------------
    # Priority 1: Title
    # --------------------------------------------------------

    title = clean_text(
        product.get("title")
    )

    if title:
        sections.append(
            ("title", f"Title: {title}")
        )

    # --------------------------------------------------------
    # Priority 2: Category
    # --------------------------------------------------------

    categories = clean_list(
        product.get("categories")
    )

    if categories:

        category_text = " > ".join(
            categories
        )

        sections.append(
            (
                "category",
                f"Category: {category_text}"
            )
        )

    # --------------------------------------------------------
    # Priority 3: Features
    # --------------------------------------------------------

    features = clean_list(
        product.get("features")
    )

    if features:

        feature_text = " | ".join(
            features
        )

        sections.append(
            (
                "features",
                f"Features: {feature_text}"
            )
        )

    # --------------------------------------------------------
    # Priority 4: Product Details
    # --------------------------------------------------------

    details_text = build_details_text(
        product.get("details")
    )

    if details_text:

        sections.append(
            (
                "details",
                f"Product Details: {details_text}"
            )
        )

    # --------------------------------------------------------
    # Priority 5: Description
    # --------------------------------------------------------

    descriptions = clean_list(
        product.get("description")
    )

    if descriptions:

        description_text = " ".join(
            descriptions
        )

        sections.append(
            (
                "description",
                f"Description: {description_text}"
            )
        )

    return sections


# ============================================================
# 7. Token 数量计算
# ============================================================

def count_tokens(text, tokenizer):
    """
    使用 BGE-M3 tokenizer 计算真实 token 数量。

    这里不截断，因为需要知道当前文本真实长度。
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False
    )

    return len(token_ids)


# ============================================================
# 8. 按 Token Budget 截断文本
# ============================================================

def truncate_to_token_budget(
    text,
    tokenizer,
    max_tokens
):
    """
    将一段文本限制在指定 token 数量以内。

    与 text[:N] 不同：
    这里根据 BGE-M3 tokenizer 的真实 token 数进行截断。
    """

    if max_tokens <= 0:
        return ""

    # --------------------------------------------------------
    # 文本 -> Token IDs
    #
    # 这里不添加 special tokens，
    # 因为最终完整 Document 才需要统一计算 special tokens。
    # --------------------------------------------------------

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    # 本身没有超过预算，直接返回
    if len(token_ids) <= max_tokens:
        return text

    # --------------------------------------------------------
    # Token-level truncation
    # --------------------------------------------------------

    token_ids = token_ids[:max_tokens]

    truncated_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True
    )

    return truncated_text.strip()


# ============================================================
# 9. 构建受 Token Budget 控制的 Product Text
# ============================================================

def build_product_text(
    product,
    tokenizer
):
    """
    根据字段优先级逐步构建 Product Document。

    核心思想：

    高优先级字段先占用 Token Budget，
    Description 最后使用剩余 Budget。
    """

    sections = build_sections(product)

    selected_sections = []

    # --------------------------------------------------------
    # 给最终模型的 special tokens 留少量空间。
    #
    # 实际最终长度仍会在最后再次验证。
    # --------------------------------------------------------

    content_budget = (
        MAX_DOCUMENT_TOKENS - 2
    )

    used_tokens = 0

    for section_name, section_text in sections:

        # ----------------------------------------------------
        # 当前 section 的 token 数
        #
        # 这里不加入 special tokens，
        # 因为每个 section 并不是独立模型输入。
        # ----------------------------------------------------

        section_tokens = tokenizer.encode(
            section_text,
            add_special_tokens=False,
            truncation=False
        )

        section_token_count = len(
            section_tokens
        )

        remaining_tokens = (
            content_budget - used_tokens
        )

        # 已经没有空间
        if remaining_tokens <= 0:
            break

        # ----------------------------------------------------
        # 当前 Section 可以完整放进去
        # ----------------------------------------------------

        if section_token_count <= remaining_tokens:

            selected_sections.append(
                section_text
            )

            used_tokens += section_token_count

            continue

        # ----------------------------------------------------
        # 当前 Section 放不下：
        #
        # 使用剩余 Token Budget 截断当前 section。
        #
        # 由于 sections 已按优先级排序，
        # 后面的低优先级字段不再继续加入。
        # ----------------------------------------------------

        truncated_section = truncate_to_token_budget(
            section_text,
            tokenizer,
            remaining_tokens
        )

        if truncated_section:

            selected_sections.append(
                truncated_section
            )

        break

    product_text = "\n".join(
        selected_sections
    )

    # --------------------------------------------------------
    # 最终保险检查
    # --------------------------------------------------------

    final_token_count = count_tokens(
        product_text,
        tokenizer
    )

    if final_token_count > MAX_DOCUMENT_TOKENS:

        product_text = truncate_to_token_budget(
            product_text,
            tokenizer,
            MAX_DOCUMENT_TOKENS - 2
        )

    return product_text


# ============================================================
# 10. Structured Metadata
# ============================================================

def build_structured_metadata(product):
    """
    保存用于过滤、排序、身份映射和最终展示的结构化属性。
    """

    return {
        "parent_asin": product.get(
            "parent_asin"
        ),
        "store": product.get(
            "store"
        ),
        "price": product.get(
            "price"
        ),
        "average_rating": product.get(
            "average_rating"
        ),
        "rating_number": product.get(
            "rating_number"
        ),
    }


# ============================================================
# 11. 构建单个 Document
# ============================================================

def build_document(
    product,
    tokenizer
):

    parent_asin = product.get(
        "parent_asin"
    )

    text = build_product_text(
        product,
        tokenizer
    )

    metadata = build_structured_metadata(
        product
    )

    return {
        "id": parent_asin,
        "text": text,
        "metadata": metadata,
    }


# ============================================================
# 12. 主流程
# ============================================================

def build_product_documents():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 加载 BGE-M3 Tokenizer
    # --------------------------------------------------------

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # 加载 10-Core Product IDs
    # --------------------------------------------------------

    print(
        "\nLoading 10-Core parent_asins..."
    )

    with open(
        TEN_CORE_ASIN_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        parent_asins = json.load(f)

    target_asins = set(
        parent_asins
    )

    print(
        f"Target Items: {len(target_asins):,}"
    )

    written_asins = set()

    written_count = 0

    # 用于观察实际发生长度控制的商品数量
    truncated_count = 0

    print(
        "\nBuilding Product Documents V2..."
    )

    # ========================================================
    # 13. Streaming Metadata Processing
    # ========================================================

    with gzip.open(
        META_PATH,
        "rt",
        encoding="utf-8"
    ) as meta_file, open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in meta_file:

            product = json.loads(line)

            parent_asin = product.get(
                "parent_asin"
            )

            # 非目标商品
            if parent_asin not in target_asins:
                continue

            # 防止重复商品
            if parent_asin in written_asins:
                continue

            # ------------------------------------------------
            # 先构建没有 Document-level Token Budget 的 sections，
            # 用于判断该商品原始 V2 文本是否超长。
            # ------------------------------------------------

            raw_sections = build_sections(
                product
            )

            raw_text = "\n".join(
                section_text
                for _, section_text in raw_sections
            )

            raw_token_count = count_tokens(
                raw_text,
                tokenizer
            )

            # ------------------------------------------------
            # 构建真正受 2048 Token Budget 控制的文档
            # ------------------------------------------------

            document = build_document(
                product,
                tokenizer
            )

            if not document["text"]:
                continue

            if raw_token_count > MAX_DOCUMENT_TOKENS:
                truncated_count += 1

            # ------------------------------------------------
            # JSONL Streaming Output
            # ------------------------------------------------

            output_file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                )
                + "\n"
            )

            written_asins.add(
                parent_asin
            )

            written_count += 1

            if written_count % 10000 == 0:

                print(
                    f"Written: {written_count:,}"
                )

            if len(written_asins) == len(target_asins):
                break

    # ========================================================
    # 14. Summary
    # ========================================================

    missing_asins = (
        target_asins - written_asins
    )

    print(
        "\n===== Product Documents V2 ====="
    )

    print(
        f"Target Items        : {len(target_asins):,}"
    )

    print(
        f"Written Documents   : {written_count:,}"
    )

    print(
        f"Missing Documents   : {len(missing_asins):,}"
    )

    print(
        f"Token Budget        : {MAX_DOCUMENT_TOKENS:,}"
    )

    print(
        f"Truncated Documents : {truncated_count:,}"
    )

    if written_count > 0:

        truncated_ratio = (
            truncated_count
            / written_count
            * 100
        )

        print(
            f"Truncated Ratio     : "
            f"{truncated_ratio:.2f}%"
        )

    print(
        f"Output Path         : {OUTPUT_PATH}"
    )

    print(
        "\nProduct Document V2 construction completed."
    )


# ============================================================
# 15. Program Entry
# ============================================================

if __name__ == "__main__":
    build_product_documents()
