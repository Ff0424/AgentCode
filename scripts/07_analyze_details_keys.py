"""
统计 10-Core 商品 Metadata 中 details 字段的 Key 分布。

目标：
1. 统计 details 中最常见的属性 Key
2. 分析 Electronics 商品常见结构化属性
3. 为后续 Product Document 的 details 字段筛选提供依据

输入：
data/processed/recommendation/parent_asins_10core.json
data/raw/meta_Electronics.jsonl.gz

当前只做统计分析，不修改原始数据。
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
# 2. 参数配置
# ============================================================

# 最终打印出现频率最高的前 100 个 details key
TOP_K = 100


# ============================================================
# 3. Details Key 分布统计
# ============================================================

def analyze_details_keys():

    # --------------------------------------------------------
    # 加载 10-Core 商品 ID
    # --------------------------------------------------------

    print("Loading 10-Core parent_asins...")

    with open(
        TEN_CORE_ASIN_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        parent_asins = json.load(f)

    target_asins = set(parent_asins)

    print(f"10-Core Items: {len(target_asins):,}")

    # --------------------------------------------------------
    # Counter:
    # key   -> 该 key 出现在多少个商品中
    #
    # 例如：
    # Brand                -> 100000
    # Compatible Devices   -> 30000
    # --------------------------------------------------------

    key_counts = Counter()

    matched_asins = set()

    print("\nScanning metadata...")

    # ========================================================
    # 4. 扫描 Metadata
    # ========================================================

    with gzip.open(
        META_PATH,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            product = json.loads(line)

            parent_asin = product.get("parent_asin")

            # 非 10-Core 商品直接跳过
            if parent_asin not in target_asins:
                continue

            matched_asins.add(parent_asin)

            details = product.get("details")

            # details 应该是 dict。
            # 如果为空或者类型异常，则跳过。
            if isinstance(details, dict) and details:

                # 一个商品中的每个 details key 计数一次
                for key in details.keys():
                    key_counts[key] += 1

            # 已经找到所有 10-Core 商品后停止扫描
            if len(matched_asins) == len(target_asins):
                break

    # ========================================================
    # 5. 输出统计结果
    # ========================================================

    print("\n===== Details Key Statistics =====")

    print(f"Matched Items       : {len(matched_asins):,}")
    print(f"Unique Details Keys : {len(key_counts):,}")

    print(f"\n===== Top {TOP_K} Details Keys =====")

    for rank, (key, count) in enumerate(
        key_counts.most_common(TOP_K),
        start=1
    ):

        ratio = (
            count / len(matched_asins) * 100
            if matched_asins
            else 0
        )

        print(
            f"{rank:>3}. "
            f"{key:<40} "
            f"{count:>8,} "
            f"({ratio:6.2f}%)"
        )


# ============================================================
# 6. Program Entry
# ============================================================

if __name__ == "__main__":
    analyze_details_keys()
