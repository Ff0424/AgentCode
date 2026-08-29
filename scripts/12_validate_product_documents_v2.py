"""
验证 Product Documents V2。

目标：
1. 验证文档数量
2. 检查是否存在空文本
3. 使用 BGE-M3 tokenizer 验证最终文档 Token 长度
4. 确认所有文档都满足 <= 2048 tokens

输入：
data/processed/product_documents_v2.jsonl
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# ============================================================
# 1. 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_documents_v2.jsonl"
)

MODEL_NAME = "BAAI/bge-m3"

MAX_DOCUMENT_TOKENS = 2048

EXPECTED_DOCUMENT_COUNT = 125_762


# ============================================================
# 2. Validation
# ============================================================

def validate_documents():

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    document_count = 0
    empty_count = 0
    over_limit_count = 0

    token_lengths = []

    # 保存最长文档的信息，方便出现问题时定位
    max_token_count = 0
    max_token_asin = None

    print("\nValidating Product Documents V2...")

    # ========================================================
    # 3. Streaming Validation
    # ========================================================

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            document = json.loads(line)

            document_count += 1

            text = document.get(
                "text",
                ""
            )

            parent_asin = document.get(
                "id"
            )

            # ------------------------------------------------
            # 检查空文本
            # ------------------------------------------------

            if not text.strip():

                empty_count += 1
                continue

            # ------------------------------------------------
            # 使用真实 BGE-M3 tokenizer 计算最终长度
            #
            # V2 理论上已经 <= 2048，
            # 因此这里不需要 truncation。
            # ------------------------------------------------

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False
            )

            token_count = len(
                token_ids
            )

            token_lengths.append(
                token_count
            )

            # ------------------------------------------------
            # 检查是否违反 Token Budget
            # ------------------------------------------------

            if token_count > MAX_DOCUMENT_TOKENS:

                over_limit_count += 1

                # 如果真的出现异常，打印前几个方便定位
                if over_limit_count <= 5:

                    print(
                        "\n[Over Limit]"
                    )

                    print(
                        f"ASIN   : {parent_asin}"
                    )

                    print(
                        f"Tokens : {token_count:,}"
                    )

            # ------------------------------------------------
            # 记录最长文档
            # ------------------------------------------------

            if token_count > max_token_count:

                max_token_count = token_count
                max_token_asin = parent_asin

            if document_count % 10000 == 0:

                print(
                    f"Validated: {document_count:,}"
                )

    # ========================================================
    # 4. Length Statistics
    # ========================================================

    token_lengths = np.array(
        token_lengths,
        dtype=np.int32
    )

    print(
        "\n===== Validation Result ====="
    )

    print(
        f"Expected Documents : "
        f"{EXPECTED_DOCUMENT_COUNT:,}"
    )

    print(
        f"Actual Documents   : "
        f"{document_count:,}"
    )

    print(
        f"Empty Documents    : "
        f"{empty_count:,}"
    )

    print(
        f"Over 2048 Tokens   : "
        f"{over_limit_count:,}"
    )

    print(
        f"Maximum Tokens     : "
        f"{max_token_count:,}"
    )

    print(
        f"Longest Document   : "
        f"{max_token_asin}"
    )

    # ========================================================
    # 5. 最终 V2 Token Distribution
    # ========================================================

    if len(token_lengths) > 0:

        print(
            "\n===== Final Token Distribution ====="
        )

        print(
            f"Mean : {token_lengths.mean():,.2f}"
        )

        print(
            f"P50  : "
            f"{np.percentile(token_lengths, 50):,.0f}"
        )

        print(
            f"P90  : "
            f"{np.percentile(token_lengths, 90):,.0f}"
        )

        print(
            f"P95  : "
            f"{np.percentile(token_lengths, 95):,.0f}"
        )

        print(
            f"P99  : "
            f"{np.percentile(token_lengths, 99):,.0f}"
        )

        print(
            f"Max  : {token_lengths.max():,.0f}"
        )

    # ========================================================
    # 6. Pass / Fail
    # ========================================================

    count_valid = (
        document_count
        == EXPECTED_DOCUMENT_COUNT
    )

    validation_passed = (
        count_valid
        and empty_count == 0
        and over_limit_count == 0
    )

    print(
        "\n===== Final Status ====="
    )

    if validation_passed:

        print(
            "PASS: Product Documents V2 "
            "passed all validation checks."
        )

    else:

        print(
            "FAIL: Product Documents V2 "
            "contains validation problems."
        )


# ============================================================
# 7. Program Entry
# ============================================================

if __name__ == "__main__":
    validate_documents()