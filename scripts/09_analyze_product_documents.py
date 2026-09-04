"""
分析已经生成的 RAG Product Documents。

目标：
1. 验证 Product Document 的实际内容
2. 统计文档字符长度和单词数量分布
3. 找出异常短 / 异常长文档
4. 为后续是否需要 Chunking、选择 Embedding Model 提供依据

输入：
data/processed/rag/product_documents.jsonl

当前只做分析，不修改 Product Documents。
"""

import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "product_documents.jsonl"
)


# ============================================================
# 2. 参数配置
# ============================================================

# 最后打印几个实际 Product Document 样本
SAMPLE_SIZE = 3

# 打印最长的几个 Product Document
TOP_LONGEST = 5


# ============================================================
# 3. Product Document 分析
# ============================================================

def analyze_product_documents():

    print("Analyzing product documents...")

    # --------------------------------------------------------
    # 保存长度统计
    #
    # 125,762 个整数的内存占用很小，
    # 所以这里没有必要为了“流式”而过度复杂化。
    # --------------------------------------------------------

    char_lengths = []
    word_lengths = []

    # 保存前几个样本，用于人工检查文档内容
    samples = []

    # 保存最长文档的信息
    # 每个元素：
    # (字符数, parent_asin, text)
    longest_documents = []

    document_count = 0

    # ========================================================
    # 4. 逐行读取 JSONL
    # ========================================================

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            document = json.loads(line)

            text = document.get("text", "")

            document_count += 1

            # ------------------------------------------------
            # 字符长度
            # ------------------------------------------------

            char_count = len(text)

            # ------------------------------------------------
            # 简单英文单词数量
            #
            # split() 按空白字符切分。
            #
            # 注意：
            # word count != token count
            #
            # 这里只用于第一阶段长度分析。
            # ------------------------------------------------

            word_count = len(text.split())

            char_lengths.append(char_count)
            word_lengths.append(word_count)

            # ------------------------------------------------
            # 保存前 SAMPLE_SIZE 个样本
            # ------------------------------------------------

            if len(samples) < SAMPLE_SIZE:
                samples.append(document)

            # ------------------------------------------------
            # 保存当前文档，用于后面寻找最长文档
            # ------------------------------------------------

            longest_documents.append(
                (
                    char_count,
                    document.get("id"),
                    text
                )
            )

    # ========================================================
    # 5. 转成 NumPy Array
    # ========================================================

    char_lengths = np.array(
        char_lengths,
        dtype=np.int32
    )

    word_lengths = np.array(
        word_lengths,
        dtype=np.int32
    )

    # ========================================================
    # 6. 基础统计
    # ========================================================

    print("\n===== Document Statistics =====")

    print(
        f"Documents : {document_count:,}"
    )

    print("\n===== Character Length =====")

    print(
        f"Min  : {char_lengths.min():,.0f}"
    )

    print(
        f"Mean : {char_lengths.mean():,.2f}"
    )

    print(
        f"P50  : {np.percentile(char_lengths, 50):,.0f}"
    )

    print(
        f"P90  : {np.percentile(char_lengths, 90):,.0f}"
    )

    print(
        f"P95  : {np.percentile(char_lengths, 95):,.0f}"
    )

    print(
        f"P99  : {np.percentile(char_lengths, 99):,.0f}"
    )

    print(
        f"Max  : {char_lengths.max():,.0f}"
    )

    print("\n===== Word Length =====")

    print(
        f"Min  : {word_lengths.min():,.0f}"
    )

    print(
        f"Mean : {word_lengths.mean():,.2f}"
    )

    print(
        f"P50  : {np.percentile(word_lengths, 50):,.0f}"
    )

    print(
        f"P90  : {np.percentile(word_lengths, 90):,.0f}"
    )

    print(
        f"P95  : {np.percentile(word_lengths, 95):,.0f}"
    )

    print(
        f"P99  : {np.percentile(word_lengths, 99):,.0f}"
    )

    print(
        f"Max  : {word_lengths.max():,.0f}"
    )

    # ========================================================
    # 7. 打印实际 Product Document 样本
    # ========================================================

    print("\n===== Product Document Samples =====")

    for index, document in enumerate(
        samples,
        start=1
    ):

        print("\n" + "=" * 80)

        print(
            f"Sample #{index}"
        )

        print(
            f"ID: {document.get('id')}"
        )

        print("-" * 80)

        print(
            document.get("text")
        )

        print("\n[Structured Metadata]")

        print(
            json.dumps(
                document.get("metadata"),
                ensure_ascii=False,
                indent=2
            )
        )

    # ========================================================
    # 8. 找出最长的 Product Documents
    # ========================================================

    longest_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print(
        f"\n===== Top {TOP_LONGEST} Longest Documents ====="
    )

    for rank, (
        char_count,
        parent_asin,
        text
    ) in enumerate(
        longest_documents[:TOP_LONGEST],
        start=1
    ):

        print("\n" + "=" * 80)

        print(
            f"Rank #{rank}"
        )

        print(
            f"ID         : {parent_asin}"
        )

        print(
            f"Characters : {char_count:,}"
        )

        print(
            f"Words      : {len(text.split()):,}"
        )

        # 这里只打印前 1000 个字符，
        # 防止终端被超长文档刷屏。
        print("\nText Preview:")

        print(
            text[:1000]
        )

        if len(text) > 1000:
            print("...")


# ============================================================
# 9. Program Entry
# ============================================================

if __name__ == "__main__":
    analyze_product_documents()
