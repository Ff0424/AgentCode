"""
Validate Recommendation -> Product Catalog -> Agent entity alignment.

Inputs:
    --item-mapping
        Recommendation item_index -> parent_asin JSON mapping.
    --product-catalog
        Canonical product_catalog.jsonl produced by V2-04.2.

Output:
    --report
        Markdown validation report for the V2-04.3 release record.

The validator never modifies the catalog. It scans the complete JSONL for
schema and identity uniqueness, then uses deterministic reservoir sampling to
validate N Recommendation-to-Agent entity mappings without loading the full
catalog into memory.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path


# ============================================================
# 1. Canonical catalog contract
# ============================================================

EXPECTED_FIELDS = (
    "item_index",
    "parent_asin",
    "title",
    "categories",
    "price",
    "average_rating",
    "rating_number",
    "features",
    "description",
)

COMPLETENESS_FIELDS = (
    "parent_asin",
    "title",
    "categories",
    "price",
    "average_rating",
    "rating_number",
)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def _is_missing(value) -> bool:
    return value is None or value == "" or value == []


def _is_optional_finite_number(value) -> bool:
    if value is None:
        return True
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_record_schema(record, row_number: int) -> list[str]:
    """Return schema/type errors for one catalog record."""

    context = f"row {row_number}"
    if not isinstance(record, dict):
        return [f"{context}: record must be a JSON object"]

    errors = []
    if tuple(record.keys()) != EXPECTED_FIELDS:
        errors.append(
            f"{context}: fields/order differ from the canonical schema"
        )

    item_index = record.get("item_index")
    if (
        isinstance(item_index, bool)
        or not isinstance(item_index, int)
        or item_index < 0
    ):
        errors.append(f"{context}: item_index must be a non-negative integer")

    parent_asin = record.get("parent_asin")
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        errors.append(f"{context}: parent_asin must be a non-empty string")

    title = record.get("title")
    if title is not None and not isinstance(title, str):
        errors.append(f"{context}: title must be a string or null")

    for field_name in ("categories", "features"):
        value = record.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            errors.append(f"{context}: {field_name} must be a list of strings")

    description = record.get("description")
    if description is not None and not isinstance(description, str):
        errors.append(f"{context}: description must be a string or null")

    for field_name in ("price", "average_rating", "rating_number"):
        if not _is_optional_finite_number(record.get(field_name)):
            errors.append(
                f"{context}: {field_name} must be a finite number or null"
            )

    rating_number = record.get("rating_number")
    if rating_number is not None and (
        not isinstance(rating_number, int)
        or isinstance(rating_number, bool)
        or rating_number < 0
    ):
        errors.append(
            f"{context}: rating_number must be a non-negative integer or null"
        )

    return errors


# ============================================================
# 2. Input loading and streaming sample
# ============================================================

def load_item_mapping(path: Path) -> dict[str, str]:
    """Load and validate the Recommendation identity mapping."""

    _require_file(path, "item mapping")
    try:
        with path.open("r", encoding="utf-8") as input_file:
            mapping = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid item mapping JSON: {exc}") from exc
    except UnicodeError as exc:
        raise ValueError(f"Item mapping is not valid UTF-8: {path}") from exc

    if not isinstance(mapping, dict):
        raise TypeError("item mapping must contain a JSON object.")
    return mapping


def scan_catalog(path: Path, sample_size: int, seed: int) -> dict:
    """Scan the full catalog and collect a deterministic reservoir sample."""

    _require_file(path, "product catalog")
    rng = random.Random(seed)
    sample = []
    seen_item_indices = set()
    seen_parent_asins = set()
    duplicate_item_indices = 0
    duplicate_parent_asins = 0
    schema_error_count = 0
    error_examples = []
    full_missing = Counter()
    row_count = 0

    try:
        with path.open("r", encoding="utf-8") as input_file:
            for row_count, line in enumerate(input_file, start=1):
                if not line.strip():
                    schema_error_count += 1
                    if len(error_examples) < 10:
                        error_examples.append(f"row {row_count}: empty JSONL line")
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    schema_error_count += 1
                    if len(error_examples) < 10:
                        error_examples.append(
                            f"row {row_count}: invalid JSON ({exc.msg})"
                        )
                    continue

                errors = validate_record_schema(record, row_count)
                schema_error_count += len(errors)
                for error in errors:
                    if len(error_examples) < 10:
                        error_examples.append(error)

                if isinstance(record, dict):
                    item_index = record.get("item_index")
                    if isinstance(item_index, int) and not isinstance(
                        item_index, bool
                    ):
                        if item_index in seen_item_indices:
                            duplicate_item_indices += 1
                        seen_item_indices.add(item_index)

                    parent_asin = record.get("parent_asin")
                    if isinstance(parent_asin, str) and parent_asin:
                        if parent_asin in seen_parent_asins:
                            duplicate_parent_asins += 1
                        seen_parent_asins.add(parent_asin)

                    for field_name in COMPLETENESS_FIELDS:
                        if _is_missing(record.get(field_name)):
                            full_missing[field_name] += 1

                # Standard reservoir sampling gives every catalog row the same
                # probability of appearing in the fixed-size validation sample.
                sample_entry = {"row_number": row_count, "record": record}
                if len(sample) < sample_size:
                    sample.append(sample_entry)
                else:
                    replacement = rng.randrange(row_count)
                    if replacement < sample_size:
                        sample[replacement] = sample_entry
    except UnicodeError as exc:
        raise ValueError(f"Product catalog is not valid UTF-8: {path}") from exc

    if row_count < sample_size:
        raise ValueError(
            f"sample_size={sample_size:,} exceeds catalog rows={row_count:,}."
        )

    return {
        "row_count": row_count,
        "sample": sample,
        "unique_item_indices": len(seen_item_indices),
        "unique_parent_asins": len(seen_parent_asins),
        "duplicate_item_indices": duplicate_item_indices,
        "duplicate_parent_asins": duplicate_parent_asins,
        "schema_error_count": schema_error_count,
        "error_examples": error_examples,
        "full_missing": full_missing,
    }


# ============================================================
# 3. Recommendation -> catalog -> Agent sample alignment
# ============================================================

def validate_sample(sample: list[dict], item_mapping: dict) -> dict:
    """Validate sampled item indices and Agent-facing product fields."""

    passed = 0
    failed = 0
    missing = Counter()
    failure_examples = []

    for sample_entry in sample:
        row_number = sample_entry["row_number"]
        record = sample_entry["record"]
        errors = validate_record_schema(record, row_number)

        if isinstance(record, dict):
            for field_name in COMPLETENESS_FIELDS:
                if _is_missing(record.get(field_name)):
                    missing[field_name] += 1

            item_index = record.get("item_index")
            parent_asin = record.get("parent_asin")
            if isinstance(item_index, int) and not isinstance(item_index, bool):
                mapped_parent_asin = item_mapping.get(str(item_index))
                if mapped_parent_asin is None:
                    errors.append(
                        f"row {row_number}: item_index is absent from item mapping"
                    )
                elif mapped_parent_asin != parent_asin:
                    errors.append(
                        f"row {row_number}: item_index maps to a different "
                        "parent_asin"
                    )

        if errors:
            failed += 1
            for error in errors:
                if len(failure_examples) < 10:
                    failure_examples.append(error)
        else:
            passed += 1

    return {
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "failure_examples": failure_examples,
    }


# ============================================================
# 4. Console and Markdown reports
# ============================================================

def print_console_report(
    sample_size: int,
    sample_result: dict,
) -> None:
    """Print the requested concise validation summary."""

    missing = sample_result["missing"]
    print("========== Product Catalog Alignment Validation ==========")
    print(f"Sample Size: {sample_size:,}")
    print(f"Passed: {sample_result['passed']:,}")
    print(f"Failed: {sample_result['failed']:,}")
    print(f"Missing title: {missing['title']:,}")
    print(f"Missing parent_asin: {missing['parent_asin']:,}")
    print(f"Missing categories: {missing['categories']:,}")
    print(f"Missing price: {missing['price']:,}")
    print(f"Missing average_rating: {missing['average_rating']:,}")
    print(f"Missing rating_number: {missing['rating_number']:,}")


def build_markdown_report(
    catalog_path: Path,
    item_mapping_path: Path,
    sample_size: int,
    seed: int,
    scan_result: dict,
    sample_result: dict,
    final_pass: bool,
) -> str:
    """Create a reproducible V2-04.3 validation report."""

    sample_missing = sample_result["missing"]
    full_missing = scan_result["full_missing"]
    status = "PASS" if final_pass else "FAIL"
    lines = [
        "# Product Catalog Alignment Validation Report",
        "",
        "## 验证目的",
        "",
        "验证 Recommendation 输出的 `item_index` 能否通过 Product Catalog "
        "稳定映射为 Agent 可使用的 `parent_asin` 和结构化商品信息。",
        "",
        "## 数据规模",
        "",
        f"- Product Catalog：`{catalog_path.as_posix()}`",
        f"- Item Mapping：`{item_mapping_path.as_posix()}`",
        f"- Catalog rows：{scan_result['row_count']:,}",
        f"- Sample size：{sample_size:,}",
        f"- Random seed：{seed}",
        "",
        "## 检查项",
        "",
        "- JSONL 每行是否为合法 JSON object；",
        "- canonical JSON schema 和字段类型是否正确；",
        "- `item_index` 与 `parent_asin` 是否全局唯一；",
        "- sampled `item_index` 是否能通过 `item_mapping.json` 映射到同一 "
        "`parent_asin`；",
        "- title、categories、price、average_rating、rating_number 的完整性；",
        "- 数值字段是否为 finite JSON numbers 或合法 null。",
        "",
        "## 验证结果",
        "",
        "### 全量 Catalog 检查",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Rows | {scan_result['row_count']:,} |",
        f"| Unique item indices | {scan_result['unique_item_indices']:,} |",
        f"| Unique parent ASINs | {scan_result['unique_parent_asins']:,} |",
        f"| Duplicate item indices | {scan_result['duplicate_item_indices']:,} |",
        f"| Duplicate parent ASINs | {scan_result['duplicate_parent_asins']:,} |",
        f"| Schema/parse errors | {scan_result['schema_error_count']:,} |",
        "",
        "### 随机样本对齐检查",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Passed | {sample_result['passed']:,} |",
        f"| Failed | {sample_result['failed']:,} |",
        f"| Missing parent_asin | {sample_missing['parent_asin']:,} |",
        f"| Missing title | {sample_missing['title']:,} |",
        f"| Missing categories | {sample_missing['categories']:,} |",
        f"| Missing price | {sample_missing['price']:,} |",
        f"| Missing average_rating | {sample_missing['average_rating']:,} |",
        f"| Missing rating_number | {sample_missing['rating_number']:,} |",
        "",
        "### 全量内容缺失统计",
        "",
        "合法的 null/空内容不属于 schema 解析错误，但在此单独记录。",
        "",
        "| Field | Missing |",
        "| --- | ---: |",
        f"| parent_asin | {full_missing['parent_asin']:,} |",
        f"| title | {full_missing['title']:,} |",
        f"| categories | {full_missing['categories']:,} |",
        f"| price | {full_missing['price']:,} |",
        f"| average_rating | {full_missing['average_rating']:,} |",
        f"| rating_number | {full_missing['rating_number']:,} |",
    ]

    errors = scan_result["error_examples"] + sample_result["failure_examples"]
    if errors:
        lines.extend(["", "### 错误示例", ""])
        lines.extend(f"- {error}" for error in errors[:10])

    lines.extend(
        [
            "",
            "## 阶段结论",
            "",
            f"**V2-04.3 Product Catalog Alignment：{status}**",
            "",
            (
                "Recommendation `item_index` 可以通过 Product Catalog "
                "稳定转换为 Agent-facing 商品实体。"
                if final_pass
                else "存在 schema、唯一性或身份映射错误，需要修复后重新验证。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report_atomic(path: Path, content: str) -> None:
    """Write the report through a temporary file before atomic publication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


# ============================================================
# 5. CLI entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Product Catalog identity and Agent field alignment."
    )
    parser.add_argument("--item-mapping", required=True, type=Path)
    parser.add_argument("--product-catalog", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if isinstance(args.sample_size, bool) or args.sample_size <= 0:
        print("ERROR: --sample-size must be a positive integer.", file=sys.stderr)
        return 2

    try:
        item_mapping = load_item_mapping(args.item_mapping)
        scan_result = scan_catalog(
            args.product_catalog, args.sample_size, args.seed
        )
        sample_result = validate_sample(scan_result["sample"], item_mapping)

        final_pass = (
            scan_result["schema_error_count"] == 0
            and scan_result["duplicate_item_indices"] == 0
            and scan_result["duplicate_parent_asins"] == 0
            and sample_result["failed"] == 0
        )
        print_console_report(args.sample_size, sample_result)
        report = build_markdown_report(
            catalog_path=args.product_catalog,
            item_mapping_path=args.item_mapping,
            sample_size=args.sample_size,
            seed=args.seed,
            scan_result=scan_result,
            sample_result=sample_result,
            final_pass=final_pass,
        )
        write_report_atomic(args.report, report)
        print(f"Report written to: {args.report}")
        print(f"FINAL STATUS: {'PASS' if final_pass else 'FAIL'}")
        return 0 if final_pass else 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
