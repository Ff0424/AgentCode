"""
抽样查看 10-Core 商品的 Metadata 内容结构。

目标：
1. 观察 title / categories / features / description / details 等字段真实内容
2. 判断哪些字段适合进入 RAG Embedding 文本
3. 判断哪些字段更适合作为结构化 Metadata 保存
4. 为后续 Product Document 构建提供依据

输入：
data/processed/parent_asins_10core.json
data/raw/meta_Electronics.jsonl.gz

说明：
这里只做抽样分析，不修改原始数据。
"""

import gzip
import json
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


# ============================================================
# 2. 参数配置
# ============================================================

# 抽样查看多少个商品
SAMPLE_SIZE = 10


# ============================================================
# 3. 格式化输出函数
# ============================================================

def print_product(product, index):
    """
    将一个商品的关键 Metadata 字段打印出来，
    方便人工观察字段结构和内容质量。
    """

    print("\n" + "=" * 80)
    print(f"Sample #{index}")
    print("=" * 80)

    print(f"parent_asin     : {product.get('parent_asin')}")
    print(f"title           : {product.get('title')}")
    print(f"store           : {product.get('store')}")
    print(f"price           : {product.get('price')}")
    print(f"average_rating  : {product.get('average_rating')}")
    print(f"rating_number   : {product.get('rating_number')}")

    print("\n[Categories]")
    print(product.get("categories"))

    print("\n[Features]")
    print(product.get("features"))

    print("\n[Description]")
    print(product.get("description"))

    print("\n[Details]")
    print(product.get("details"))


# ============================================================
# 4. 抽样 10-Core Metadata
# ============================================================

def explore_metadata():

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

    # 转换为 set，方便快速判断当前 Metadata 是否属于目标商品
    target_asins = set(parent_asins)

    print(f"10-Core Items: {len(target_asins):,}")
    print(f"Sample Size  : {SAMPLE_SIZE}")

    # --------------------------------------------------------
    # 从 Metadata 文件中找到前 SAMPLE_SIZE 个 10-Core 商品
    # --------------------------------------------------------

    samples = []

    print("\nScanning metadata...")

    with gzip.open(
        META_PATH,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            product = json.loads(line)

            parent_asin = product.get("parent_asin")

            # 不属于 10-Core 商品，直接跳过
            if parent_asin not in target_asins:
                continue

            samples.append(product)

            # 已经收集够样本后停止扫描
            if len(samples) >= SAMPLE_SIZE:
                break

    # ========================================================
    # 5. 打印样本
    # ========================================================

    print(f"\nCollected {len(samples)} samples.")

    for index, product in enumerate(samples, start=1):
        print_product(product, index)


# ============================================================
# 6. Program Entry
# ============================================================

if __name__ == "__main__":
    explore_metadata()