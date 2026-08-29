"""
Amazon Electronics Interaction 数据预处理
JSON Review → Integer Interaction + Mapping
功能：
1. 读取 reviews_Electronics.jsonl.gz
2. 提取 user_id 和 parent_asin
3. 将字符串 ID 映射为连续整数 ID
4. 保存整数化后的 User-Item Interactions
5. 保存整数 ID 与 Amazon 原始 ID 的映射关系

输出：
data/processed/
├── interactions.npz
├── user_mapping.json
└── item_mapping.json
"""

import gzip
import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

# 当前脚本位于 scripts/ 目录，因此 parent.parent 为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reviews_Electronics.jsonl.gz"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INTERACTION_PATH = PROCESSED_DIR / "interactions.npz"
USER_MAPPING_PATH = PROCESSED_DIR / "user_mapping.json"
ITEM_MAPPING_PATH = PROCESSED_DIR / "item_mapping.json"


# ============================================================
# 2. 读取 Review，并构建整数 ID
# ============================================================

def prepare_interactions():

    # --------------------------------------------------------
    # Mapping：
    # Amazon 原始字符串 ID -> 连续整数 ID
    #
    # 例如：
    # "AFKZ..."    -> 0
    # "B083NRGZMM" -> 0
    # --------------------------------------------------------

    user_mapping = {}
    item_mapping = {}

    # 保存每一条 interaction 对应的整数 User / Item ID
    user_ids = []
    item_ids = []

    print("Reading Amazon Electronics reviews...")

    with gzip.open(REVIEW_PATH, "rt", encoding="utf-8") as f:

        for index, line in enumerate(f, start=1):

            review = json.loads(line)

            raw_user_id = review["user_id"]
            raw_item_id = review["parent_asin"]

            # ------------------------------------------------
            # 第一次遇到某个 User 时，为其分配新的整数 ID
            # ------------------------------------------------
            if raw_user_id not in user_mapping:
                user_mapping[raw_user_id] = len(user_mapping)

            # ------------------------------------------------
            # 第一次遇到某个 Item 时，为其分配新的整数 ID
            # ------------------------------------------------
            if raw_item_id not in item_mapping:
                item_mapping[raw_item_id] = len(item_mapping)

            user_ids.append(user_mapping[raw_user_id])
            item_ids.append(item_mapping[raw_item_id])

            # 每处理 100 万条打印一次进度
            if index % 1_000_000 == 0:
                print(
                    f"Processed {index:,} interactions | "
                    f"Users={len(user_mapping):,} | "
                    f"Items={len(item_mapping):,}"
                )

    # ========================================================
    # 3. Python List -> NumPy int32 Array
    # ========================================================

    # 连续整数 ID 的规模远小于 int32 上限，
    # 使用 int32 可以显著减少存储和内存占用。
    user_ids = np.asarray(user_ids, dtype=np.int32)
    item_ids = np.asarray(item_ids, dtype=np.int32)

    # 如果 processed/ 不存在，则自动创建
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 4. 保存 User-Item Interaction
    # ========================================================

    np.savez(
        INTERACTION_PATH,
        user_ids=user_ids,
        item_ids=item_ids,
    )

    print(f"\nInteractions saved to:")
    print(INTERACTION_PATH)

    # ========================================================
    # 5. 构建反向 Mapping
    #
    # 当前：
    # Amazon ID -> Integer ID
    #
    # 后续推荐系统真正需要：
    # Integer ID -> Amazon ID
    #
    # 例如：
    # 952 -> "B083NRGZMM"
    # ========================================================

    id_to_user = {
        integer_id: raw_id
        for raw_id, integer_id in user_mapping.items()
    }

    id_to_item = {
        integer_id: raw_id
        for raw_id, integer_id in item_mapping.items()
    }

    # ========================================================
    # 6. 保存 Mapping
    # ========================================================

    with open(USER_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(
            id_to_user,
            f,
            ensure_ascii=False
        )

    with open(ITEM_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(
            id_to_item,
            f,
            ensure_ascii=False
        )

    # ========================================================
    # 7. 输出最终统计信息
    # ========================================================

    print("\n===== Finished =====")

    print(f"Interactions : {len(user_ids):,}")
    print(f"Users        : {len(user_mapping):,}")
    print(f"Items        : {len(item_mapping):,}")

    print("\nMapping files:")

    print(f"User mapping : {USER_MAPPING_PATH}")
    print(f"Item mapping : {ITEM_MAPPING_PATH}")


# ============================================================
# 8. Program Entry
# ============================================================

if __name__ == "__main__":
    prepare_interactions()