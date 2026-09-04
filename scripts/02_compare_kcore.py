"""
K-Core 数据过滤策略对比实验（原始 Baseline 实现）。

目标：
1. 在 Amazon Electronics 原始 Review 数据上执行 K-Core Filtering
2. 比较不同 K 值过滤后的 User / Item / Interaction 数量
3. 理解 K-Core 必须迭代执行的原因
4. 作为后续 NumPy 优化版本的 Baseline

输入：
data/raw/reviews_Electronics.jsonl.gz

输出：
无数据文件。

说明：
本脚本每轮 K-Core 都会重新扫描并解压完整 Review 文件，
因此在 4388 万条交互上运行速度较慢。

后续的 03_compare_kcore_fast.py 会使用预处理后的整数 ID
和 NumPy bincount / boolean mask 进行加速。

本脚本主要用于理解算法原理和结果验证，
不作为后续正式数据处理 Pipeline。
"""

import gzip
import json
from collections import Counter
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

REVIEW_PATH = (
    RAW_DIR
    / "reviews_Electronics.jsonl.gz"
)


# ============================================================
# 2. 统计当前子图中的 User / Item Degree
# ============================================================

def count_degrees(
    valid_users=None,
    valid_items=None
):
    """
    扫描 Review 数据，统计当前有效子图中：

    1. 每个 User 的交互次数（degree）
    2. 每个 Item 的交互次数（degree）
    3. 当前子图的 Interaction 数量

    valid_users / valid_items 为 None 时，
    表示第一次迭代，使用完整数据。
    """

    user_counts = Counter()
    item_counts = Counter()

    interaction_count = 0

    with gzip.open(
        REVIEW_PATH,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            review = json.loads(line)

            user_id = review["user_id"]
            item_id = review["parent_asin"]

            # ------------------------------------------------
            # 如果已经有有效 User 集合，
            # 则过滤掉上一轮被删除的 User。
            # ------------------------------------------------

            if (
                valid_users is not None
                and user_id not in valid_users
            ):
                continue

            # ------------------------------------------------
            # 同理，过滤上一轮被删除的 Item。
            # ------------------------------------------------

            if (
                valid_items is not None
                and item_id not in valid_items
            ):
                continue

            user_counts[user_id] += 1
            item_counts[item_id] += 1

            interaction_count += 1

    return (
        user_counts,
        item_counts,
        interaction_count
    )


# ============================================================
# 3. Iterative K-Core Filtering
# ============================================================

def run_kcore(k):
    """
    执行 K-Core Filtering。

    K-Core 要求最终保留下来的：

    每个 User 至少有 K 条交互
    每个 Item 至少有 K 条交互

    注意：
    不能只过滤一次。

    因为删除低频 User 后，
    Item 的 Degree 可能下降；

    删除低频 Item 后，
    User 的 Degree 也可能下降。

    因此需要不断迭代直到 User / Item 集合稳定。
    """

    valid_users = None
    valid_items = None

    iteration = 0

    while True:

        iteration += 1

        # ----------------------------------------------------
        # 根据上一轮保留下来的 User / Item，
        # 重新扫描数据并计算 Degree。
        # ----------------------------------------------------

        (
            user_counts,
            item_counts,
            interaction_count
        ) = count_degrees(
            valid_users,
            valid_items
        )

        # ----------------------------------------------------
        # 保留 Degree >= K 的 User
        # ----------------------------------------------------

        new_valid_users = {
            user_id
            for user_id, count
            in user_counts.items()
            if count >= k
        }

        # ----------------------------------------------------
        # 保留 Degree >= K 的 Item
        # ----------------------------------------------------

        new_valid_items = {
            item_id
            for item_id, count
            in item_counts.items()
            if count >= k
        }

        print(
            f"K={k}, "
            f"iteration={iteration}, "
            f"users={len(new_valid_users):,}, "
            f"items={len(new_valid_items):,}, "
            f"interactions={interaction_count:,}"
        )

        # ----------------------------------------------------
        # 如果 User / Item 集合都不再发生变化，
        # 说明 K-Core 已经收敛。
        # ----------------------------------------------------

        if (
            new_valid_users == valid_users
            and new_valid_items == valid_items
        ):
            break

        valid_users = new_valid_users
        valid_items = new_valid_items

    return (
        len(valid_users),
        len(valid_items),
        interaction_count
    )


# ============================================================
# 4. 比较不同 K 值
# ============================================================

results = {}

# 当前保留 5-Core / 10-Core 对比。
#
# 如果需要重新完整比较：
# K_VALUES = [3, 5, 10]
K_VALUES = [5, 10]


for k in K_VALUES:

    print(
        f"\n===== Running {k}-Core ====="
    )

    (
        users,
        items,
        interactions
    ) = run_kcore(k)

    results[k] = {
        "users": users,
        "items": items,
        "interactions": interactions,
    }


# ============================================================
# 5. 输出最终结果
# ============================================================

print(
    "\n===== Final Results ====="
)

for k, result in results.items():

    print(
        f"{k}-Core: "
        f"Users={result['users']:,}, "
        f"Items={result['items']:,}, "
        f"Interactions="
        f"{result['interactions']:,}"
    )