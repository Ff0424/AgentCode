"""
Amazon Electronics 原始 Review 数据 EDA。

目标：
1. 统计 Review / User / Item 数量
2. 分析 Rating 分布
3. 分析 User / Item 交互次数
4. 分析交互次数分位数
5. 分析 User / Item 长尾分布

输入：
data/raw/reviews_Electronics.jsonl.gz

输出：
无数据文件。

说明：
本脚本只对原始 Review 数据进行探索性数据分析（EDA），
统计结果直接打印到终端，不修改原始数据，
也不向 data/processed/ 写入任何文件。
"""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

# 当前脚本：
# AgentCode/scripts/01_analyze_reviews.py
#
# parent      -> scripts/
# parent.parent -> AgentCode/
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
# 2. 基础统计容器
# ============================================================

review_count = 0

# set 用于统计不重复的 User / Item 数量
users = set()
items = set()

# Counter 用于统计 Rating 频次
rating_counts = Counter()

# Counter 用于统计每个 User / Item 的交互次数
user_counts = Counter()
item_counts = Counter()


# ============================================================
# 3. Streaming 读取 Amazon Review 数据
# ============================================================

print(f"Reading reviews from: {REVIEW_PATH}")

with gzip.open(
    REVIEW_PATH,
    "rt",
    encoding="utf-8"
) as f:

    for line in f:

        review = json.loads(line)

        review_count += 1

        user_id = review["user_id"]
        item_id = review["parent_asin"]
        rating = review["rating"]

        # ----------------------------------------------------
        # 统计唯一 User / Item
        # ----------------------------------------------------

        users.add(user_id)
        items.add(item_id)

        # ----------------------------------------------------
        # Rating 分布
        # ----------------------------------------------------

        rating_counts[rating] += 1

        # ----------------------------------------------------
        # User / Item 交互频次
        # ----------------------------------------------------

        user_counts[user_id] += 1
        item_counts[item_id] += 1


# ============================================================
# 4. Dataset 基础统计
# ============================================================

print("\n===== Dataset Statistics =====")

print("Review count:", review_count)
print("Unique users:", len(users))
print("Unique items:", len(items))


# ============================================================
# 5. Rating Distribution
# ============================================================

print("\n===== Rating Distribution =====")

for rating in sorted(rating_counts):

    print(
        rating,
        ":",
        rating_counts[rating]
    )


# ============================================================
# 6. User / Item Interaction Statistics
# ============================================================

print("\n===== User Interaction Statistics =====")

print(
    "Min:",
    min(user_counts.values())
)

print(
    "Max:",
    max(user_counts.values())
)

print(
    "Average:",
    sum(user_counts.values())
    / len(user_counts)
)


print("\n===== Item Interaction Statistics =====")

print(
    "Min:",
    min(item_counts.values())
)

print(
    "Max:",
    max(item_counts.values())
)

print(
    "Average:",
    sum(item_counts.values())
    / len(item_counts)
)


# ============================================================
# 7. Interaction Percentiles
# ============================================================

user_values = list(
    user_counts.values()
)

item_values = list(
    item_counts.values()
)


print("\n===== User Interaction Percentiles =====")

print(
    "P50:",
    np.percentile(user_values, 50)
)

print(
    "P90:",
    np.percentile(user_values, 90)
)

print(
    "P95:",
    np.percentile(user_values, 95)
)

print(
    "P99:",
    np.percentile(user_values, 99)
)


print("\n===== Item Interaction Percentiles =====")

print(
    "P50:",
    np.percentile(item_values, 50)
)

print(
    "P90:",
    np.percentile(item_values, 90)
)

print(
    "P95:",
    np.percentile(item_values, 95)
)

print(
    "P99:",
    np.percentile(item_values, 99)
)


# ============================================================
# 8. Interaction Frequency Distribution
# ============================================================

def count_frequency_distribution(counts):
    """
    将 User / Item 的交互次数划分到不同区间。

    例如：

    User A -> 1 次
    User B -> 1 次
    User C -> 5 次

    最终：

    "1" -> 2
    "5" -> 1

    用于观察数据集中的长尾现象。
    """

    distribution = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        ">100": 0,
    }

    for count in counts.values():

        if count == 1:
            distribution["1"] += 1

        elif count == 2:
            distribution["2"] += 1

        elif count == 3:
            distribution["3"] += 1

        elif count == 4:
            distribution["4"] += 1

        elif count == 5:
            distribution["5"] += 1

        elif count <= 10:
            distribution["6-10"] += 1

        elif count <= 20:
            distribution["11-20"] += 1

        elif count <= 50:
            distribution["21-50"] += 1

        elif count <= 100:
            distribution["51-100"] += 1

        else:
            distribution[">100"] += 1

    return distribution


# ============================================================
# 9. User / Item Long-Tail Distribution
# ============================================================

user_distribution = (
    count_frequency_distribution(
        user_counts
    )
)

item_distribution = (
    count_frequency_distribution(
        item_counts
    )
)


print(
    "\n===== User Interaction Frequency Distribution ====="
)

for group, count in user_distribution.items():

    print(
        f"{group}: {count}"
    )


print(
    "\n===== Item Interaction Frequency Distribution ====="
)

for group, count in item_distribution.items():

    print(
        f"{group}: {count}"
    )