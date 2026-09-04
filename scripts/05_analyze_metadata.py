"""
分析 10-Core 商品的 Amazon Metadata 覆盖率和字段完整度

目标：
1. 检查 10-Core 商品有多少能在 Metadata 中找到
2. 分析关键 Metadata 字段的完整程度
3. 为后续 RAG Product Document 构建提供依据

输入：
data/processed/recommendation/parent_asins_10core.json
data/raw/meta_Electronics.jsonl.gz

当前只做统计分析，不修改原始数据，也不生成最终 RAG 数据。
"""

import gzip
import json
from collections import Counter
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
    / "recommendation"
    / "parent_asins_10core.json"
)


# ============================================================
# 2. 判断 Metadata 字段是否有效
# ============================================================

def is_available(value):
    """
    判断一个 Metadata 字段是否包含有效信息。

    以下情况视为缺失：
    - None
    - 空字符串 ""
    - 空 list []
    - 空 dict {}
    """

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    if isinstance(value, (list, dict)):
        return len(value) > 0

    # 数字、布尔值等非空对象默认认为有效
    return True


# ============================================================
# 3. Metadata Coverage Analysis
# ============================================================

def analyze_metadata():

    # --------------------------------------------------------
    # 加载 10-Core 商品 parent_asin
    # --------------------------------------------------------

    print("Loading 10-Core parent_asins...")

    with open(
        TEN_CORE_ASIN_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        parent_asins = json.load(f)

    # 转成 set，提高 membership lookup 速度
    target_asins = set(parent_asins)

    print(f"10-Core Items: {len(target_asins):,}")

    # --------------------------------------------------------
    # 我们重点关注这些 Metadata 字段
    # --------------------------------------------------------

    fields = [
        "title",
        "description",
        "features",
        "categories",
        "price",
        "store",
        "details",
        "average_rating",
        "rating_number",
    ]

    field_counts = Counter()

    matched_asins = set()

    total_metadata = 0

    print("\nScanning metadata...")

    # ========================================================
    # 4. 扫描 Amazon Metadata
    # ========================================================

    with gzip.open(
        META_PATH,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            total_metadata += 1

            product = json.loads(line)

            parent_asin = product.get("parent_asin")

            # 当前商品不属于 10-Core，则直接跳过
            if parent_asin not in target_asins:
                continue

            matched_asins.add(parent_asin)

            # ------------------------------------------------
            # 统计各字段是否包含有效内容
            # ------------------------------------------------

            for field in fields:

                value = product.get(field)

                if is_available(value):
                    field_counts[field] += 1

            # ------------------------------------------------
            # 所有目标商品都已经找到，可以提前结束扫描
            # ------------------------------------------------

            if len(matched_asins) == len(target_asins):
                break

    # ========================================================
    # 5. Coverage Statistics
    # ========================================================

    total_target = len(target_asins)
    matched_count = len(matched_asins)

    missing_count = total_target - matched_count

    coverage = (
        matched_count / total_target * 100
        if total_target > 0
        else 0
    )

    print("\n===== Metadata Coverage =====")

    print(f"10-Core Items     : {total_target:,}")
    print(f"Matched Metadata  : {matched_count:,}")
    print(f"Missing Metadata  : {missing_count:,}")
    print(f"Coverage          : {coverage:.2f}%")

    # ========================================================
    # 6. Field Availability Statistics
    # ========================================================

    print("\n===== Field Availability =====")

    for field in fields:

        count = field_counts[field]

        ratio = (
            count / matched_count * 100
            if matched_count > 0
            else 0
        )

        print(
            f"{field:<16} "
            f"{count:>8,} / {matched_count:,} "
            f"({ratio:6.2f}%)"
        )

    # ========================================================
    # 7. Missing ASIN Examples
    # ========================================================

    missing_asins = target_asins - matched_asins

    if missing_asins:

        print("\n===== Missing ASIN Examples =====")

        # 只打印前 10 个，避免输出过多
        for asin in list(missing_asins)[:10]:
            print(asin)

    # ========================================================
    # 8. Final Summary
    # ========================================================

    print("\n===== Finished =====")

    print(
        f"Scanned {total_metadata:,} metadata records."
    )


# ============================================================
# 9. Program Entry
# ============================================================

if __name__ == "__main__":
    analyze_metadata()
