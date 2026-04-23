from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Dict, Iterable, List, Sequence

from .data_types import TrainingLedgerEntry


def stable_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merkle_root(leaves: Sequence[str]) -> str:
    if not leaves:
        return sha256_hex("")
    level = [sha256_hex(leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: List[str] = []
        for idx in range(0, len(level), 2):
            next_level.append(sha256_hex(level[idx] + level[idx + 1]))
        level = next_level
    return level[0]


def commit_parameters(weights: Sequence[float], bias: float, nonce: str | None = None) -> Dict[str, str]:
    nonce = nonce or secrets.token_hex(16)
    rounded = [round(value, 8) for value in weights]
    commitment = sha256_hex(stable_dumps({"weights": rounded, "bias": round(bias, 8), "nonce": nonce}))
    return {"nonce": nonce, "commitment": commitment}


class ProofOfTrainingLedger:
    """
    Research-prototype PoT layer.

    This is not a zk-SNARK implementation; it preserves the paper's structure by
    recording immutable commitments, Merkle roots, and a hash-chained training log.
    """

    def __init__(self, train_doc_ids: Sequence[str]) -> None:
        self.entries: List[TrainingLedgerEntry] = []
        self.dataset_root = merkle_root(list(train_doc_ids))
        self.initial_commitment = ""
        self.final_commitment = ""

    def start(self, weights: Sequence[float], bias: float) -> None:
        self.initial_commitment = commit_parameters(weights, bias)["commitment"]

    def append_step(
        self,
        step: int,
        batch_doc_ids: Sequence[str],
        metrics: Dict[str, float],
        weights: Sequence[float],
        bias: float,
    ) -> None:
        parameter_commitment = commit_parameters(weights, bias)["commitment"]
        self.entries.append(
            TrainingLedgerEntry(
                step=step,
                batch_doc_ids=list(batch_doc_ids),
                metric_snapshot={key: round(value, 6) for key, value in metrics.items()},
                parameter_commitment=parameter_commitment,
            )
        )
        self.final_commitment = parameter_commitment

    def fingerprint(self) -> Dict[str, Any]:
        hash_chain = []
        previous = self.initial_commitment or sha256_hex("empty")
        for entry in self.entries:
            current = sha256_hex(previous + stable_dumps(entry.to_dict()))
            hash_chain.append({"step": entry.step, "digest": current})
            previous = current
        return {
            "dataset_root": self.dataset_root,
            "initial_commitment": self.initial_commitment,
            "final_commitment": self.final_commitment,
            "entry_count": len(self.entries),
            "hash_chain_tail": hash_chain[-1]["digest"] if hash_chain else self.initial_commitment,
            "entries": [entry.to_dict() for entry in self.entries],
        }
