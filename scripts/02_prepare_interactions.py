"""
Amazon Electronics Interaction 数据预处理。

JSON Review
    ↓
Integer Interaction
    +
ID Mapping

功能：
1. 读取 reviews_Electronics.jsonl.gz
2. 提取 user_id 和 parent_asin
3. 将 Amazon 字符串 ID 映射为连续整数 ID
4. 保存整数化后的 User-Item Interactions
5. 保存整数 ID 与 Amazon 原始 ID 的映射关系

输入：
data/raw/reviews_Electronics.jsonl.gz

输出：
data/processed/recommendation/
├── interactions.npz
├── user_mapping.json
└── item_mapping.json

说明：
这些文件属于 Recommendation 数据链路，
因此统一保存到 processed/recommendation/。
"""

import gzip
import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

# 当前脚本：
# AgentCode/scripts/02_prepare_interactions.py
#
# parent.parent -> AgentCode/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

# Recommendation 相关处理结果统一放在该目录
RECOMMENDATION_DIR = (
    PROCESSED_DIR
    / "recommendation"
)


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

REVIEW_PATH = (
    RAW_DIR
    / "reviews_Electronics.jsonl.gz"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

INTERACTION_PATH = (
    RECOMMENDATION_DIR
    / "interactions.npz"
)

USER_MAPPING_PATH = (
    RECOMMENDATION_DIR
    / "user_mapping.json"
)

ITEM_MAPPING_PATH = (
    RECOMMENDATION_DIR
    / "item_mapping.json"
)


# ============================================================
# 2. 读取 Review，并构建整数 ID
# ============================================================

def prepare_interactions():

    # --------------------------------------------------------
    # Mapping：
    #
    # Amazon 原始字符串 ID
    #       ↓
    # 连续整数 ID
    #
    # 例如：
    #
    # User:
    # "AFKZ..." -> 0
    #
    # Item:
    # "B083NRGZMM" -> 0
    # --------------------------------------------------------

    user_mapping = {}
    item_mapping = {}

    # --------------------------------------------------------
    # 保存每一条 Interaction 对应的整数 User / Item ID
    #
    # 例如：
    #
    # user_ids = [0, 1, 0, 2, ...]
    # item_ids = [3, 5, 8, 3, ...]
    #
    # 两个数组相同位置共同表示一条交互：
    #
    # user_ids[i] -> item_ids[i]
    # --------------------------------------------------------

    user_ids = []
    item_ids = []

    print("Reading Amazon Electronics reviews...")

    # ========================================================
    # 3. Streaming 读取 Review
    # ========================================================

    with gzip.open(
        REVIEW_PATH,
        "rt",
        encoding="utf-8"
    ) as f:

        for index, line in enumerate(
            f,
            start=1
        ):

            review = json.loads(line)

            raw_user_id = review["user_id"]
            raw_item_id = review["parent_asin"]

            # ------------------------------------------------
            # 第一次遇到某个 User 时，
            # 为其分配一个新的连续整数 ID。
            # ------------------------------------------------

            if raw_user_id not in user_mapping:

                user_mapping[raw_user_id] = (
                    len(user_mapping)
                )

            # ------------------------------------------------
            # 第一次遇到某个 Item 时，
            # 为其分配一个新的连续整数 ID。
            # ------------------------------------------------

            if raw_item_id not in item_mapping:

                item_mapping[raw_item_id] = (
                    len(item_mapping)
                )

            # ------------------------------------------------
            # 保存当前 Interaction
            # ------------------------------------------------

            user_ids.append(
                user_mapping[raw_user_id]
            )

            item_ids.append(
                item_mapping[raw_item_id]
            )

            # ------------------------------------------------
            # 每处理 100 万条打印一次进度
            # ------------------------------------------------

            if index % 1_000_000 == 0:

                print(
                    f"Processed {index:,} interactions | "
                    f"Users={len(user_mapping):,} | "
                    f"Items={len(item_mapping):,}"
                )


    # ========================================================
    # 4. Python List -> NumPy int32 Array
    # ========================================================

    # 当前 User / Item 数量远小于 int32 上限。
    #
    # 相比 int64：
    # int32 每个整数只占 4 Bytes，
    # 可以降低内存和磁盘占用。

    user_ids = np.asarray(
        user_ids,
        dtype=np.int32
    )

    item_ids = np.asarray(
        item_ids,
        dtype=np.int32
    )


    # ========================================================
    # 5. 创建 Recommendation 输出目录
    # ========================================================

    RECOMMENDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 6. 保存 User-Item Interactions
    # ========================================================

    np.savez(
        INTERACTION_PATH,
        user_ids=user_ids,
        item_ids=item_ids,
    )

    print(
        "\nInteractions saved to:"
    )

    print(
        INTERACTION_PATH
    )


    # ========================================================
    # 7. 构建反向 Mapping
    # ========================================================

    # 上面构建 Mapping 时使用的是：
    #
    # Amazon ID -> Integer ID
    #
    # 例如：
    #
    # "B083NRGZMM" -> 952
    #
    # 但是推荐模型以后输出的是 Integer ID。
    #
    # 因此真正需要保存：
    #
    # Integer ID -> Amazon ID
    #
    # 例如：
    #
    # 952 -> "B083NRGZMM"

    id_to_user = {
        integer_id: raw_id
        for raw_id, integer_id
        in user_mapping.items()
    }

    id_to_item = {
        integer_id: raw_id
        for raw_id, integer_id
        in item_mapping.items()
    }


    # ========================================================
    # 8. 保存 User Mapping
    # ========================================================

    with open(
        USER_MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            id_to_user,
            f,
            ensure_ascii=False
        )


    # ========================================================
    # 9. 保存 Item Mapping
    # ========================================================

    with open(
        ITEM_MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            id_to_item,
            f,
            ensure_ascii=False
        )


    # ========================================================
    # 10. 输出最终统计信息
    # ========================================================

    print(
        "\n===== Finished ====="
    )

    print(
        f"Interactions : {len(user_ids):,}"
    )

    print(
        f"Users        : {len(user_mapping):,}"
    )

    print(
        f"Items        : {len(item_mapping):,}"
    )

    print(
        "\nOutput Files:"
    )

    print(
        f"Interactions : {INTERACTION_PATH}"
    )

    print(
        f"User mapping : {USER_MAPPING_PATH}"
    )

    print(
        f"Item mapping : {ITEM_MAPPING_PATH}"
    )


# ============================================================
# 11. Program Entry
# ============================================================

if __name__ == "__main__":
    prepare_interactions()