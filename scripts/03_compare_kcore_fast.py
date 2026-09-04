"""
High-performance K-Core filtering using NumPy.
3/5/10-Core 对比实验

输入：
data/processed/recommendation/interactions.npz
"""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation"
    / "interactions.npz"
)


def run_kcore(user_ids, item_ids, num_users, num_items, k):
    active = np.ones(len(user_ids), dtype=bool)

    iteration = 0

    while True:
        iteration += 1

        active_users = user_ids[active]
        active_items = item_ids[active]

        user_counts = np.bincount(
            active_users,
            minlength=num_users
        )

        item_counts = np.bincount(
            active_items,
            minlength=num_items
        )

        valid_users = user_counts >= k
        valid_items = item_counts >= k

        new_active = (
            active
            & valid_users[user_ids]
            & valid_items[item_ids]
        )

        interaction_count = int(new_active.sum())
        user_count = int(valid_users.sum())
        item_count = int(valid_items.sum())

        print(
            f"K={k}, iteration={iteration}, "
            f"users={user_count:,}, "
            f"items={item_count:,}, "
            f"interactions={interaction_count:,}"
        )

        if np.array_equal(new_active, active):
            break

        active = new_active

    return user_count, item_count, interaction_count


def main():
    print("Loading interactions...")

    data = np.load(DATA_PATH)

    user_ids = data["user_ids"]
    item_ids = data["item_ids"]

    num_users = int(user_ids.max()) + 1
    num_items = int(item_ids.max()) + 1

    print(f"Interactions: {len(user_ids):,}")
    print(f"Users: {num_users:,}")
    print(f"Items: {num_items:,}")

    results = {}

    for k in [3, 5, 10]:
        print(f"\n===== Running {k}-Core =====")

        users, items, interactions = run_kcore(
            user_ids,
            item_ids,
            num_users,
            num_items,
            k,
        )

        results[k] = {
            "users": users,
            "items": items,
            "interactions": interactions,
        }

    print("\n===== Final Results =====")

    for k, result in results.items():
        print(
            f"{k}-Core: "
            f"Users={result['users']:,}, "
            f"Items={result['items']:,}, "
            f"Interactions={result['interactions']:,}"
        )


if __name__ == "__main__":
    main()
