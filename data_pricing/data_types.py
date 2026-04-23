from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentRecord:
    """Canonical training or validation sample."""

    doc_id: str
    source_id: str
    text: str
    label: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentRecord":
        return cls(
            doc_id=str(payload["doc_id"]),
            source_id=str(payload["source_id"]),
            text=str(payload["text"]),
            label=int(payload.get("label", 0)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "text": self.text,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class DocumentScore:
    doc_id: str
    source_id: str
    token_count: int
    info_density_bits: float
    normalized_info_density: float
    syntactic_coherence: float
    semantic_richness: float
    dqs: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "token_count": self.token_count,
            "info_density_bits": self.info_density_bits,
            "normalized_info_density": self.normalized_info_density,
            "syntactic_coherence": self.syntactic_coherence,
            "semantic_richness": self.semantic_richness,
            "dqs": self.dqs,
        }


@dataclass
class SourceScore:
    source_id: str
    document_count: int
    token_count: int
    mean_dqs: float
    proxy_gain: float
    influence_score: float
    shapley_value: float
    unified_score: float
    confidence_low: float
    confidence_high: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "document_count": self.document_count,
            "token_count": self.token_count,
            "mean_dqs": self.mean_dqs,
            "proxy_gain": self.proxy_gain,
            "influence_score": self.influence_score,
            "shapley_value": self.shapley_value,
            "unified_score": self.unified_score,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


@dataclass
class TrainingLedgerEntry:
    step: int
    batch_doc_ids: List[str]
    metric_snapshot: Dict[str, float]
    parameter_commitment: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "batch_doc_ids": self.batch_doc_ids,
            "metric_snapshot": self.metric_snapshot,
            "parameter_commitment": self.parameter_commitment,
        }


@dataclass
class ValuationReport:
    document_scores: List[DocumentScore]
    source_scores: List[SourceScore]
    ledger: Dict[str, Any]
    config: Dict[str, Any]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_scores": [item.to_dict() for item in self.document_scores],
            "source_scores": [item.to_dict() for item in self.source_scores],
            "ledger": self.ledger,
            "config": self.config,
            "summary": self.summary,
        }
