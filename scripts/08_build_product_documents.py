"""
构建 10-Core 商品的 RAG Product Documents。

目标：
1. 从 Amazon Electronics Metadata 中提取 10-Core 商品
2. 将适合语义检索的字段构造成 text
3. 将价格、评分等字段保存为 structured metadata
4. 输出 JSONL 文件，为后续 Embedding / Vector Store 做准备

输入：
data/processed/parent_asins_10core.json
data/raw/meta_Electronics.jsonl.gz

输出：
data/processed/product_documents.jsonl

每行格式：
{
    "id": "...",
    "text": "...",
    "metadata": {...}
}
"""

import gzip
import json
import re
from pathlib import Path


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
    / "parent_asins_10core.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_documents.jsonl"
)


# ============================================================
# 2. Product Details 过滤配置
# ============================================================

# 这些字段主要属于 Amazon 页面管理信息，
# 对 V1 商品语义检索价值相对较低，因此暂时不放入 Embedding Text。
EXCLUDED_DETAIL_KEYS = {
    "Date First Available",
    "Best Sellers Rank",
    "Is Discontinued By Manufacturer",
    "Part Number",
    "Manufacturer Part Number",
}


# 单个 Details value 最大保留字符数。
# 防止某些异常字段过长，导致一个属性占据大量文本。
MAX_DETAIL_VALUE_LENGTH = 500


# ============================================================
# 3. 基础文本清洗
# ============================================================

def clean_text(value):
    """
    对文本进行轻量清洗。

    当前 V1 只处理：
    1. None
    2. 多余空白字符
    3. Amazon 页面中的 "See more" 残留

    不进行复杂 NLP 清洗，避免误删商品语义。
    """

    if value is None:
        return ""

    # 保证输入统一转换为字符串
    text = str(value)

    # 去掉 Amazon 页面中常见的 "See more" 残留
    text = re.sub(
        r"\s*See more\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 将连续空格、换行、Tab 等统一成一个空格
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. List 类型字段转换
# ============================================================

def clean_list(values):
    """
    清洗 Amazon Metadata 中的 list 类型文本字段。

    例如：
    features = [
        "Bluetooth 5.0",
        "100W Power Delivery"
    ]

    返回：
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
# 5. 构建 Product Details 文本
# ============================================================

def build_details_text(details):
    """
    将 details dict 转换成适合 Product Document 的文本。

    示例：

    {
        "Brand": "PNY",
        "Memory Storage Capacity": "1 TB"
    }

    转换为：

    Brand: PNY
    Memory Storage Capacity: 1 TB
    """

    if not isinstance(details, dict):
        return []

    detail_lines = []

    for key, value in details.items():

        # ----------------------------------------------------
        # 过滤低价值管理字段
        # ----------------------------------------------------

        if key in EXCLUDED_DETAIL_KEYS:
            continue

        cleaned_key = clean_text(key)
        cleaned_value = clean_text(value)

        # key 或 value 无有效内容则跳过
        if not cleaned_key or not cleaned_value:
            continue

        # ----------------------------------------------------
        # 防止异常超长属性进入 Product Document
        # ----------------------------------------------------

        if len(cleaned_value) > MAX_DETAIL_VALUE_LENGTH:
            cleaned_value = (
                cleaned_value[:MAX_DETAIL_VALUE_LENGTH]
                + "..."
            )

        detail_lines.append(
            f"{cleaned_key}: {cleaned_value}"
        )

    return detail_lines


# ============================================================
# 6. 构建用于 Embedding 的 Product Text
# ============================================================

def build_product_text(product):
    """
    将 Amazon Metadata 转换成用于后续语义检索的文本。

    当前 V1 使用：
    - Title
    - Category
    - Features
    - Description
    - Product Details

    注意：
    price / rating 等结构化字段不在这里处理，
    它们会单独进入 metadata。
    """

    sections = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = clean_text(product.get("title"))

    if title:
        sections.append(
            f"Title: {title}"
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    categories = clean_list(
        product.get("categories")
    )

    if categories:

        category_text = " > ".join(categories)

        sections.append(
            f"Category: {category_text}"
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = clean_list(
        product.get("features")
    )

    if features:

        feature_text = " | ".join(features)

        sections.append(
            f"Features: {feature_text}"
        )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    descriptions = clean_list(
        product.get("description")
    )

    if descriptions:

        description_text = " ".join(descriptions)

        sections.append(
            f"Description: {description_text}"
        )

    # --------------------------------------------------------
    # Product Details
    # --------------------------------------------------------

    detail_lines = build_details_text(
        product.get("details")
    )

    if detail_lines:

        details_text = " | ".join(detail_lines)

        sections.append(
            f"Product Details: {details_text}"
        )

    # 不同 section 使用换行分隔，
    # 保留 Product Document 的基本结构。
    return "\n".join(sections)


# ============================================================
# 7. 构建 Structured Metadata
# ============================================================

def build_structured_metadata(product):
    """
    保存不主要依赖 Embedding 处理的结构化字段。

    这些字段以后可用于：
    - 商品身份映射
    - 数值过滤
    - 排序
    - Agent 最终回答展示
    """

    return {
        "parent_asin": product.get("parent_asin"),
        "store": product.get("store"),
        "price": product.get("price"),
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
    }


# ============================================================
# 8. 构建单个 Product Document
# ============================================================

def build_document(product):
    """
    将单条 Amazon Metadata 转换成统一的 Product Document。
    """

    parent_asin = product.get("parent_asin")

    text = build_product_text(product)

    metadata = build_structured_metadata(product)

    return {
        "id": parent_asin,
        "text": text,
        "metadata": metadata,
    }


# ============================================================
# 9. 扫描 Metadata 并生成 Product Documents
# ============================================================

def build_product_documents():

    # --------------------------------------------------------
    # 加载 10-Core 商品集合
    # --------------------------------------------------------

    print("Loading 10-Core parent_asins...")

    with open(
        TEN_CORE_ASIN_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        parent_asins = json.load(f)

    target_asins = set(parent_asins)

    print(f"Target Items: {len(target_asins):,}")

    matched_asins = set()

    written_count = 0

    print("\nBuilding product documents...")

    # --------------------------------------------------------
    # 同时打开输入 Metadata 和输出 JSONL
    # --------------------------------------------------------

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

            parent_asin = product.get("parent_asin")

            # 非 10-Core 商品直接跳过
            if parent_asin not in target_asins:
                continue

            # ------------------------------------------------
            # 防止 Metadata 中出现重复 parent_asin 时重复写入
            # ------------------------------------------------

            if parent_asin in matched_asins:
                continue

            document = build_document(product)

            # 如果 text 完全为空，则没有可检索商品知识
            if not document["text"]:
                continue

            # ------------------------------------------------
            # 每个 Product Document 写成一行 JSON
            # ------------------------------------------------

            output_file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                )
                + "\n"
            )

            matched_asins.add(parent_asin)

            written_count += 1

            # 每处理 10,000 个商品打印一次进度
            if written_count % 10000 == 0:
                print(
                    f"Written: {written_count:,}"
                )

            # 已找到全部目标商品，可以停止扫描
            if len(matched_asins) == len(target_asins):
                break

    # ========================================================
    # 10. 最终统计
    # ========================================================

    missing_asins = (
        target_asins - matched_asins
    )

    print("\n===== Product Document Statistics =====")

    print(
        f"Target Items     : {len(target_asins):,}"
    )

    print(
        f"Written Documents: {written_count:,}"
    )

    print(
        f"Missing Documents: {len(missing_asins):,}"
    )

    print(
        f"Output Path      : {OUTPUT_PATH}"
    )

    print("\nProduct document construction completed.")


# ============================================================
# 11. Program Entry
# ============================================================

if __name__ == "__main__":
    build_product_documents()