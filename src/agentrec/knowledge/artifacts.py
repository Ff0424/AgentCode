"""Deterministic, fingerprinted builder for read-only product knowledge files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .schemas import KnowledgeChunk, KnowledgeChunkType, ProductDocument


ARTIFACT_VERSION = "agentrec-knowledge-v1"
OUTPUT_FILES = (
    "product_documents.jsonl", "chunks.jsonl", "chunk_metadata.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical item mapping not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("Canonical item mapping must be a JSON object.")
    result: dict[int, str] = {}
    for raw_index, raw_asin in value.items():
        if not isinstance(raw_index, str) or not raw_index.isdecimal():
            raise ValueError(f"Invalid canonical item index key: {raw_index!r}.")
        if not isinstance(raw_asin, str) or not raw_asin.strip():
            raise ValueError(f"Invalid parent_asin for item_index={raw_index}.")
        result[int(raw_index)] = raw_asin.strip()
    return result


def _parts(text: str, max_chars: int) -> tuple[str, ...]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and size + addition > max_chars:
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + (1 if size else 0)
    if current:
        parts.append(" ".join(current))
    return tuple(parts)


def _document(record: object, line_number: int, mapping: dict[int, str]) -> ProductDocument:
    if not isinstance(record, dict):
        raise TypeError(f"Catalog line {line_number} must be a JSON object.")
    document = ProductDocument.model_validate(record)
    expected = mapping.get(document.item_index)
    if expected is None:
        raise ValueError(f"Catalog item_index={document.item_index} is absent from mapping.")
    if expected != document.parent_asin:
        raise ValueError(f"Canonical identity mismatch at Catalog line {line_number}.")
    return document


def _chunk_texts(document: ProductDocument) -> tuple[tuple[KnowledgeChunkType, str], ...]:
    summary = []
    if document.title:
        summary.append(f"Title: {document.title}")
    if document.categories:
        summary.append(f"Categories: {' | '.join(document.categories)}")
    if document.price is not None:
        summary.append(f"Price: {document.price:g}")
    if document.average_rating is not None:
        rating = f"Average rating: {document.average_rating:g}"
        if document.rating_number is not None:
            rating += f" ({document.rating_number} ratings)"
        summary.append(rating)
    if not summary:
        summary.append(f"Parent ASIN: {document.parent_asin}")
    values = [(KnowledgeChunkType.SUMMARY, "\n".join(summary))]
    if document.features:
        values.append((KnowledgeChunkType.FEATURES, "\n".join(document.features)))
    if document.description:
        values.append((KnowledgeChunkType.DESCRIPTION, document.description))
    return tuple(values)


def build_knowledge_artifacts(
    *, catalog_path: str | Path, item_mapping_path: str | Path,
    output_dir: str | Path, max_chunk_chars: int = 1600,
) -> dict:
    """Build and atomically publish a complete knowledge directory."""

    catalog = Path(catalog_path).resolve()
    mapping_path = Path(item_mapping_path).resolve()
    destination = Path(output_dir).resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"Product Catalog not found: {catalog}")
    if isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int) or max_chunk_chars < 100:
        raise ValueError("max_chunk_chars must be an integer >= 100.")
    if destination.exists():
        raise FileExistsError(f"Knowledge output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapping = _load_mapping(mapping_path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        documents_path = temporary / OUTPUT_FILES[0]
        chunks_path = temporary / OUTPUT_FILES[1]
        seen_items: set[int] = set()
        seen_asins: set[str] = set()
        chunk_count = 0
        type_counts: Counter[str] = Counter()
        with (
            catalog.open("r", encoding="utf-8") as source,
            documents_path.open("w", encoding="utf-8", newline="\n") as documents,
            chunks_path.open("w", encoding="utf-8", newline="\n") as chunks,
        ):
            for line_number, line in enumerate(source, 1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid Catalog JSON at line {line_number}.") from exc
                document = _document(raw, line_number, mapping)
                if document.item_index in seen_items or document.parent_asin in seen_asins:
                    raise ValueError(f"Duplicate Catalog identity at line {line_number}.")
                seen_items.add(document.item_index)
                seen_asins.add(document.parent_asin)
                documents.write(document.model_dump_json() + "\n")
                for chunk_type, text in _chunk_texts(document):
                    for part_index, part in enumerate(_parts(text, max_chunk_chars)):
                        chunk = KnowledgeChunk(
                            chunk_id=f"{document.item_index}:{chunk_type.value}:{part_index}",
                            chunk_index=chunk_count, item_index=document.item_index,
                            parent_asin=document.parent_asin, chunk_type=chunk_type,
                            part_index=part_index, text=part,
                        )
                        chunks.write(chunk.model_dump_json() + "\n")
                        chunk_count += 1
                        type_counts[chunk_type.value] += 1
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "product_count": len(seen_items), "chunk_count": chunk_count,
            "chunk_type_counts": dict(sorted(type_counts.items())),
            "max_chunk_chars": max_chunk_chars,
            "identity_contract": "chunk.parent_asin -> chunk.item_index -> canonical mapping",
        }
        (temporary / OUTPUT_FILES[2]).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "sources": {
                "product_catalog": {"path": str(catalog), "sha256": _sha256(catalog)},
                "item_mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
            },
            "outputs": {
                name: {"sha256": _sha256(temporary / name)} for name in OUTPUT_FILES
            },
            "product_count": len(seen_items), "chunk_count": chunk_count,
        }
        (temporary / "knowledge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_knowledge_artifacts(output_dir: str | Path) -> dict:
    """Fail fast when a published knowledge file no longer matches its manifest."""

    root = Path(output_dir).resolve()
    manifest_path = root / "knowledge_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Knowledge manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported knowledge artifact version.")
    for name in OUTPUT_FILES:
        path = root / name
        expected = manifest.get("outputs", {}).get(name, {}).get("sha256")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Knowledge artifact fingerprint mismatch: {name}")
    return manifest
