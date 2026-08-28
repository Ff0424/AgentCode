"""
数据过滤策略实验
3-Core / 5-Core / 10-Core 对比实验
"""

import gzip
import json
from collections import Counter
from pathlib import Path


REVIEW_PATH = Path("../data/raw/reviews_Electronics.jsonl.gz")


def count_degrees(valid_users=None, valid_items=None):
    user_counts = Counter()
    item_counts = Counter()
    interaction_count = 0

    with gzip.open(REVIEW_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)

            user_id = review["user_id"]
            item_id = review["parent_asin"]

            if valid_users is not None and user_id not in valid_users:
                continue

            if valid_items is not None and item_id not in valid_items:
                continue

            user_counts[user_id] += 1
            item_counts[item_id] += 1
            interaction_count += 1

    return user_counts, item_counts, interaction_count

def run_kcore(k):
    valid_users = None
    valid_items = None

    iteration = 0

    while True:
        iteration += 1

        user_counts, item_counts, interaction_count = count_degrees(
            valid_users,
            valid_items
        )

        new_valid_users = {
            user_id
            for user_id, count in user_counts.items()
            if count >= k
        }

        new_valid_items = {
            item_id
            for item_id, count in item_counts.items()
            if count >= k
        }

        print(
            f"K={k}, iteration={iteration}, "
            f"users={len(new_valid_users)}, "
            f"items={len(new_valid_items)}, "
            f"interactions={interaction_count}"
        )

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

results = {}

for k in [5,10]:
# for k in [3, 5, 10]:
    print(f"\n===== Running {k}-Core =====")

    users, items, interactions = run_kcore(k)

    results[k] = {
        "users": users,
        "items": items,
        "interactions": interactions,
    }


print("\n===== Final Results =====")

for k, result in results.items():
    print(
        f"{k}-Core: "
        f"Users={result['users']}, "
        f"Items={result['items']}, "
        f"Interactions={result['interactions']}"
    )