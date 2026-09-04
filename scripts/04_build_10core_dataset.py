"""
根据实验结论正式生成 10-Core 数据
构建 Amazon Electronics 10-Core 推荐数据集

功能：
1. 加载整数化后的 User-Item Interactions
2. 执行 10-Core Filtering
3. 保存最终保留的 Interaction
4. 保存最终保留的 User / Item 整数 ID
5. 将 Item 整数 ID 转换回 Amazon parent_asin
6. 为后续 Metadata 匹配做准备

输入：
data/processed/recommendation/
├── interactions.npz
└── item_mapping.json

输出：
data/processed/recommendation/
├── interactions_10core.npz
├── users_10core.npy
├── items_10core.npy
└── parent_asins_10core.json
"""

import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECOMMENDATION_DIR = (
    PROJECT_ROOT / "data" / "processed" / "recommendation"
)

INTERACTION_PATH = RECOMMENDATION_DIR / "interactions.npz"
ITEM_MAPPING_PATH = RECOMMENDATION_DIR / "item_mapping.json"

OUTPUT_INTERACTION_PATH = (
    RECOMMENDATION_DIR / "interactions_10core.npz"
)

OUTPUT_USER_PATH = (
    RECOMMENDATION_DIR / "users_10core.npy"
)

OUTPUT_ITEM_PATH = (
    RECOMMENDATION_DIR / "items_10core.npy"
)

OUTPUT_PARENT_ASIN_PATH = (
    RECOMMENDATION_DIR / "parent_asins_10core.json"
)


# ============================================================
# 2. K-Core Filtering
# ============================================================

def run_kcore(user_ids, item_ids, num_users, num_items, k):
    """
    对 User-Item Interaction 执行迭代式 K-Core Filtering。

    返回：
        active:
            Boolean Mask。
            True 表示该 interaction 最终被保留。
    """

    # 初始状态：所有 interaction 都有效
    active = np.ones(len(user_ids), dtype=bool)

    iteration = 0

    while True:
        iteration += 1

        # ----------------------------------------------------
        # 取出当前仍然有效的 interaction
        # ----------------------------------------------------

        active_users = user_ids[active]
        active_items = item_ids[active]

        # ----------------------------------------------------
        # 统计当前子图中每个 User / Item 的 degree
        # ----------------------------------------------------

        user_counts = np.bincount(
            active_users,
            minlength=num_users
        )

        item_counts = np.bincount(
            active_items,
            minlength=num_items
        )

        # degree >= K 的节点才能继续保留
        valid_users = user_counts >= k
        valid_items = item_counts >= k

        # ----------------------------------------------------
        # 一条 interaction 只有在：
        #
        # 1. 原本仍然有效
        # 2. User 满足 K-Core
        # 3. Item 满足 K-Core
        #
        # 三个条件同时满足时才能保留
        # ----------------------------------------------------

        new_active = (
            active
            & valid_users[user_ids]
            & valid_items[item_ids]
        )

        interaction_count = int(new_active.sum())

        # 注意：
        # 这里统计最终实际参与 new_active 的 User / Item，
        # 用于输出当前迭代结果。
        remaining_users = np.unique(
            user_ids[new_active]
        )

        remaining_items = np.unique(
            item_ids[new_active]
        )

        print(
            f"K={k}, iteration={iteration}, "
            f"users={len(remaining_users):,}, "
            f"items={len(remaining_items):,}, "
            f"interactions={interaction_count:,}"
        )

        # Mask 不再变化，说明 K-Core 已经收敛
        if np.array_equal(new_active, active):
            break

        active = new_active

    return active


# ============================================================
# 3. 构建正式 10-Core Dataset
# ============================================================

def build_10core_dataset():

    RECOMMENDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading interactions...")

    data = np.load(INTERACTION_PATH)

    user_ids = data["user_ids"]
    item_ids = data["item_ids"]

    num_users = int(user_ids.max()) + 1
    num_items = int(item_ids.max()) + 1

    print(f"Raw interactions : {len(user_ids):,}")
    print(f"Raw users        : {num_users:,}")
    print(f"Raw items        : {num_items:,}")

    # ========================================================
    # 4. 执行 10-Core
    # ========================================================

    print("\n===== Running 10-Core =====")

    active = run_kcore(
        user_ids=user_ids,
        item_ids=item_ids,
        num_users=num_users,
        num_items=num_items,
        k=10,
    )

    # ========================================================
    # 5. 提取最终 10-Core Interactions
    # ========================================================

    filtered_user_ids = user_ids[active]
    filtered_item_ids = item_ids[active]

    unique_users = np.unique(filtered_user_ids)
    unique_items = np.unique(filtered_item_ids)

    print("\n===== Final 10-Core Dataset =====")

    print(f"Users        : {len(unique_users):,}")
    print(f"Items        : {len(unique_items):,}")
    print(f"Interactions : {len(filtered_user_ids):,}")

    # ========================================================
    # 6. 保存 10-Core Interaction
    # ========================================================

    np.savez(
        OUTPUT_INTERACTION_PATH,
        user_ids=filtered_user_ids,
        item_ids=filtered_item_ids,
    )

    # 保存 10-Core 中实际存在的整数 User / Item ID
    np.save(
        OUTPUT_USER_PATH,
        unique_users
    )

    np.save(
        OUTPUT_ITEM_PATH,
        unique_items
    )

    # ========================================================
    # 7. Integer Item ID -> Amazon parent_asin
    # ========================================================

    print("\nLoading item mapping...")

    with open(
        ITEM_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        item_mapping = json.load(f)

    # JSON Object 的 key 会被读取为字符串，
    # 因此这里需要使用 str(item_id) 查询。
    parent_asins = [
        item_mapping[str(int(item_id))]
        for item_id in unique_items
    ]

    # ========================================================
    # 8. 保存 10-Core parent_asin
    # ========================================================

    with open(
        OUTPUT_PARENT_ASIN_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            parent_asins,
            f,
            ensure_ascii=False
        )

    # ========================================================
    # 9. 完成
    # ========================================================

    print("\n===== Saved Files =====")

    print(OUTPUT_INTERACTION_PATH)
    print(OUTPUT_USER_PATH)
    print(OUTPUT_ITEM_PATH)
    print(OUTPUT_PARENT_ASIN_PATH)

    print("\n10-Core dataset construction completed.")


# ============================================================
# 10. Program Entry
# ============================================================

if __name__ == "__main__":
    build_10core_dataset()
