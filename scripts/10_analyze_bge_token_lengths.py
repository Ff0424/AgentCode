"""
使用 BGE-M3 官方 tokenizer 分析 Product Document 的 Token 长度。

目标：
1. 使用真实 tokenizer，而不是 word count 估计模型输入长度
2. 统计 125,762 个商品的 token 长度分布
3. 统计超过不同长度阈值的商品数量
4. 为 Product Document V2 的长度控制策略提供依据

输入：
data/processed/rag/product_documents.jsonl

模型：
BAAI/bge-m3

注意：
这里只加载 tokenizer，不加载完整 Embedding Model。
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# ============================================================
# 1. 路径与模型配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "product_documents.jsonl"
)

MODEL_NAME = "BAAI/bge-m3"


# ============================================================
# 2. Token 长度阈值
# ============================================================

# 这些阈值用于观察 Product Document 的长尾情况。
# 8192 是 BGE-M3 官方支持的最大输入长度。
TOKEN_THRESHOLDS = [
    512,
    1024,
    2048,
    4096,
    8192,
]


# ============================================================
# 3. Token Length Analysis
# ============================================================

def analyze_token_lengths():

    # --------------------------------------------------------
    # 加载 BGE-M3 tokenizer
    #
    # 注意：
    # AutoTokenizer 只负责把文本转换成 token IDs，
    # 这里不会加载完整的 BGE-M3 Embedding 模型。
    # --------------------------------------------------------

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    token_lengths = []

    document_count = 0

    print("\nAnalyzing token lengths...")

    # ========================================================
    # 4. 逐行读取 Product Documents
    # ========================================================

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            document = json.loads(line)

            text = document.get("text", "")

            # ------------------------------------------------
            # tokenize，但不做 truncation。
            #
            # 我们现在的目的就是观察原始文档究竟有多长，
            # 如果这里提前 truncation=True，
            # 就无法发现超过模型限制的文档。
            # ------------------------------------------------

            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,  # 分词时不要把超长文本截断。
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            token_count = len(
                encoded["input_ids"]
            )

            token_lengths.append(
                token_count
            )

            document_count += 1

            if document_count % 10000 == 0:

                print(
                    f"Processed: {document_count:,}"
                )

    # ========================================================
    # 5. 转换成 NumPy Array
    # ========================================================

    token_lengths = np.array(
        token_lengths,
        dtype=np.int32
    )

    # ========================================================
    # 6. Token Length Distribution
    # ========================================================

    print("\n===== Token Length Statistics =====")

    print(
        f"Documents : {document_count:,}"
    )

    print(
        f"Min  : {token_lengths.min():,.0f}"
    )

    print(
        f"Mean : {token_lengths.mean():,.2f}"
    )

    print(
        f"P50  : {np.percentile(token_lengths, 50):,.0f}"
    )

    print(
        f"P90  : {np.percentile(token_lengths, 90):,.0f}"
    )

    print(
        f"P95  : {np.percentile(token_lengths, 95):,.0f}"
    )

    print(
        f"P99  : {np.percentile(token_lengths, 99):,.0f}"
    )

    print(
        f"Max  : {token_lengths.max():,.0f}"
    )

    # ========================================================
    # 7. Threshold Analysis
    # ========================================================

    print("\n===== Token Threshold Analysis =====")

    for threshold in TOKEN_THRESHOLDS:

        count = int(
            np.sum(
                token_lengths > threshold
            )
        )

        ratio = (
            count
            / document_count
            * 100
        )

        print(
            f"> {threshold:>4} tokens : "
            f"{count:>7,} "
            f"({ratio:6.2f}%)"
        )


# ============================================================
# 8. Program Entry
# ============================================================

if __name__ == "__main__":
    analyze_token_lengths()
