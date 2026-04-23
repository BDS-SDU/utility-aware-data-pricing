from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .data_types import DocumentRecord


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for a real public fine-tuning dataset."""

    name: str
    domain: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    citation_url: str
    license_note: str
    formatter: Callable[[Dict[str, Any]], str]


def format_alpaca(row: Dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{output}"
    return f"Instruction:\n{instruction}\n\nResponse:\n{output}"


def format_gsm8k(row: Dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    return f"Math problem:\n{question}\n\nSolution:\n{answer}"


def format_codexglue_python(row: Dict[str, Any]) -> str:
    docstring = str(row.get("docstring", "")).strip()
    code = str(row.get("code", row.get("original_string", ""))).strip()
    func_name = str(row.get("func_name", "")).strip()
    return f"Function:\n{func_name}\n\nDocstring:\n{docstring}\n\nCode:\n{code}"


REAL_DATASET_SPECS: Dict[str, DatasetSpec] = {
    "alpaca_instruction": DatasetSpec(
        name="alpaca_instruction",
        domain="general_instruction",
        hf_path="tatsu-lab/alpaca",
        hf_config=None,
        split="train",
        citation_url="https://huggingface.co/datasets/tatsu-lab/alpaca",
        license_note="CC BY-NC 4.0; model-generated instruction-following data.",
        formatter=format_alpaca,
    ),
    "gsm8k_math": DatasetSpec(
        name="gsm8k_math",
        domain="math_reasoning",
        hf_path="openai/gsm8k",
        hf_config="main",
        split="train",
        citation_url="https://huggingface.co/datasets/openai/gsm8k",
        license_note="GSM8K grade-school math word problems.",
        formatter=format_gsm8k,
    ),
    "codexglue_python": DatasetSpec(
        name="codexglue_python",
        domain="code_summarization",
        hf_path="google/code_x_glue_ct_code_to_text",
        hf_config="python",
        split="train",
        citation_url="https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text",
        license_note="C-UDA licensed CodeXGLUE/CodeSearchNet code-to-text data.",
        formatter=format_codexglue_python,
    ),
}


def _read_local_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _local_dataset_path(local_data_dir: Path, spec: DatasetSpec) -> Optional[Path]:
    candidates = [
        local_data_dir / f"{spec.name}.jsonl",
        local_data_dir / f"{spec.name}_{spec.split}.jsonl",
        local_data_dir / spec.name / f"{spec.split}.jsonl",
        local_data_dir / spec.name / "data.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_local_rows(spec: DatasetSpec, local_data_dir: Path, limit: int) -> List[Dict[str, Any]]:
    path = _local_dataset_path(local_data_dir, spec)
    if path is None:
        expected = ", ".join(
            [
                f"{spec.name}.jsonl",
                f"{spec.name}_{spec.split}.jsonl",
                f"{spec.name}/{spec.split}.jsonl",
                f"{spec.name}/data.jsonl",
            ]
        )
        raise FileNotFoundError(
            f"Cannot find local data for {spec.name} under {local_data_dir}. "
            f"Expected one of: {expected}."
        )
    return _read_local_jsonl(path, limit)


def _load_huggingface_rows(
    spec: DatasetSpec,
    limit: int,
    seed: int,
    local_files_only: bool = False,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The real multi-domain experiment requires the optional Hugging Face "
            "`datasets` package. Install it with `pip install datasets`."
        ) from exc

    try:
        dataset = (
            load_dataset(
                spec.hf_path,
                spec.hf_config,
                split=spec.split,
                streaming=not local_files_only,
                download_mode="reuse_dataset_if_exists",
            )
            if spec.hf_config
            else load_dataset(
                spec.hf_path,
                split=spec.split,
                streaming=not local_files_only,
                download_mode="reuse_dataset_if_exists",
            )
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face"
        raise RuntimeError(
            f"Failed to load {spec.name} from {mode}. If this machine cannot access "
            "huggingface.co, either run once on a machine with network/cache access, "
            "set a mirror with HF_ENDPOINT, or pass --local-data-dir with JSONL files. "
            f"Original error: {exc}"
        ) from exc
    if local_files_only:
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            if idx >= limit:
                break
            rows.append(dict(row))
        return rows
    shuffled = dataset.shuffle(buffer_size=max(1_000, limit * 10), seed=seed)
    rows: List[Dict[str, Any]] = []
    for row in shuffled.take(limit):
        rows.append(dict(row))
    return rows


def load_real_domain_records(
    spec: DatasetSpec,
    train_limit: int,
    val_limit: int,
    shards: int,
    seed: int,
    local_data_dir: Optional[Path] = None,
    hf_local_files_only: bool = False,
) -> Dict[str, List[DocumentRecord]]:
    if local_data_dir is not None:
        rows = _load_local_rows(spec, local_data_dir, train_limit + val_limit)
    else:
        rows = _load_huggingface_rows(
            spec,
            train_limit + val_limit,
            seed,
            local_files_only=hf_local_files_only,
        )
    records: List[DocumentRecord] = []
    for idx, row in enumerate(rows):
        text = spec.formatter(row)
        if not text.strip():
            continue
        split = "val" if idx < val_limit else "train"
        shard_idx = (idx - val_limit) % max(1, shards)
        source_id = f"{spec.name}_shard_{shard_idx:02d}" if split == "train" else f"{spec.name}_validation"
        records.append(
            DocumentRecord(
                doc_id=f"{spec.name}-{split}-{idx}",
                source_id=source_id,
                text=text,
                label=1,
                metadata={
                    "domain": spec.domain,
                    "dataset": spec.name,
                    "split": split,
                    "source_url": spec.citation_url,
                    "license_note": spec.license_note,
                },
            )
        )
    return {
        "train": [record for record in records if record.metadata["split"] == "train"],
        "val": [record for record in records if record.metadata["split"] == "val"],
    }


def build_target_validation(
    target_domain: str,
    domain_validation: Dict[str, List[DocumentRecord]],
    negatives_per_domain: int,
) -> List[DocumentRecord]:
    validation: List[DocumentRecord] = []
    for domain, records in domain_validation.items():
        limit = len(records) if domain == target_domain else min(negatives_per_domain, len(records))
        for record in records[:limit]:
            label = 1 if domain == target_domain else 0
            validation.append(
                DocumentRecord(
                    doc_id=f"{target_domain}-eval-{record.doc_id}",
                    source_id="validation",
                    text=record.text,
                    label=label,
                    metadata={
                        **record.metadata,
                        "target_domain": target_domain,
                        "eval_weight": 1.0 if label == 1 else 0.35,
                    },
                )
            )
    return validation


def load_all_real_domains(
    dataset_names: Iterable[str],
    train_limit_per_domain: int = 120,
    val_limit_per_domain: int = 40,
    shards_per_domain: int = 4,
    seed: int = 17,
    local_data_dir: Optional[str | Path] = None,
    hf_local_files_only: bool = False,
) -> Dict[str, Dict[str, List[DocumentRecord]]]:
    output: Dict[str, Dict[str, List[DocumentRecord]]] = {}
    local_path = Path(local_data_dir) if local_data_dir is not None else None
    for offset, dataset_name in enumerate(dataset_names):
        spec = REAL_DATASET_SPECS[dataset_name]
        output[spec.domain] = load_real_domain_records(
            spec=spec,
            train_limit=train_limit_per_domain,
            val_limit=val_limit_per_domain,
            shards=shards_per_domain,
            seed=seed + offset,
            local_data_dir=local_path,
            hf_local_files_only=hf_local_files_only,
        )
    return output
