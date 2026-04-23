from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from .data_types import DocumentRecord
from .quality import simple_tokenize, stable_hash_index


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class HashedTextLogReg:
    """Pure Python proxy model for fast valuation experiments."""

    def __init__(
        self,
        dim: int = 256,
        lr: float = 0.15,
        epochs: int = 12,
        l2: float = 1e-3,
        seed: int = 7,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights = [0.0] * dim
        self.bias = 0.0

    def featurize(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        tokens = simple_tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            vector[stable_hash_index(token, self.dim)] += 1.0
        scale = 1.0 / len(tokens)
        return [value * scale for value in vector]

    def _score(self, features: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def predict_proba_from_features(self, features: Sequence[float]) -> float:
        return sigmoid(self._score(features))

    def fit(self, records: Sequence[DocumentRecord]) -> None:
        randomizer = random.Random(self.seed)
        rows = [(self.featurize(record.text), record.label) for record in records]
        for _ in range(self.epochs):
            randomizer.shuffle(rows)
            for features, label in rows:
                pred = self.predict_proba_from_features(features)
                error = pred - label
                for idx, value in enumerate(features):
                    self.weights[idx] -= self.lr * (error * value + self.l2 * self.weights[idx])
                self.bias -= self.lr * error

    def evaluate(self, records: Sequence[DocumentRecord]) -> Dict[str, float]:
        if not records:
            return {"accuracy": 0.0, "log_loss": 0.0, "value": 0.0}
        weighted_correct = 0.0
        total_weight = 0.0
        total_loss = 0.0
        positive_score = 0.0
        negative_score = 0.0
        positive_weight = 0.0
        negative_weight = 0.0
        for record in records:
            pred = self.predict_proba_from_features(self.featurize(record.text))
            label = record.label
            weight = float(record.metadata.get("eval_weight", 1.0))
            clipped = min(max(pred, 1e-8), 1.0 - 1e-8)
            total_loss += weight * (-(label * math.log(clipped) + (1 - label) * math.log(1 - clipped)))
            weighted_correct += weight * int((pred >= 0.5) == bool(label))
            total_weight += weight
            if label == 1:
                positive_score += weight * pred
                positive_weight += weight
            else:
                negative_score += weight * pred
                negative_weight += weight
        accuracy = weighted_correct / max(total_weight, 1e-12)
        log_loss = total_loss / max(total_weight, 1e-12)
        positive_mean = positive_score / max(positive_weight, 1e-12)
        negative_mean = negative_score / max(negative_weight, 1e-12)
        task_utility = positive_mean - 0.5 * negative_mean
        return {
            "accuracy": accuracy,
            "log_loss": log_loss,
            "value": task_utility - log_loss,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
        }

    def average_gradient(self, records: Sequence[DocumentRecord]) -> Tuple[List[float], float]:
        if not records:
            return [0.0] * self.dim, 0.0
        grad_w = [0.0] * self.dim
        grad_b = 0.0
        for record in records:
            features = self.featurize(record.text)
            pred = self.predict_proba_from_features(features)
            error = pred - record.label
            for idx, value in enumerate(features):
                grad_w[idx] += error * value
            grad_b += error
        scale = 1.0 / len(records)
        return [value * scale for value in grad_w], grad_b * scale

    def hessian_vector_product(
        self,
        records: Sequence[DocumentRecord],
        vector_w: Sequence[float],
        vector_b: float,
    ) -> Tuple[List[float], float]:
        out_w = [self.l2 * value for value in vector_w]
        out_b = 0.0
        if not records:
            return out_w, out_b
        scale = 1.0 / len(records)
        for record in records:
            features = self.featurize(record.text)
            pred = self.predict_proba_from_features(features)
            curvature = pred * (1.0 - pred)
            direction = sum(f * v for f, v in zip(features, vector_w)) + vector_b
            factor = curvature * direction * scale
            for idx, value in enumerate(features):
                out_w[idx] += factor * value
            out_b += factor
        return out_w, out_b


def dot(left_w: Sequence[float], left_b: float, right_w: Sequence[float], right_b: float) -> float:
    return sum(a * b for a, b in zip(left_w, right_w)) + left_b * right_b


def conjugate_gradient(
    matvec: Callable[[Sequence[float], float], Tuple[List[float], float]],
    rhs_w: Sequence[float],
    rhs_b: float,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> Tuple[List[float], float]:
    x_w = [0.0] * len(rhs_w)
    x_b = 0.0
    r_w = list(rhs_w)
    r_b = rhs_b
    p_w = list(r_w)
    p_b = r_b
    rs_old = dot(r_w, r_b, r_w, r_b)
    if rs_old == 0.0:
        return x_w, x_b
    for _ in range(max_iter):
        hp_w, hp_b = matvec(p_w, p_b)
        denom = max(dot(p_w, p_b, hp_w, hp_b), 1e-12)
        alpha = rs_old / denom
        for idx in range(len(x_w)):
            x_w[idx] += alpha * p_w[idx]
            r_w[idx] -= alpha * hp_w[idx]
        x_b += alpha * p_b
        r_b -= alpha * hp_b
        rs_new = dot(r_w, r_b, r_w, r_b)
        if math.sqrt(rs_new) < tol:
            break
        beta = rs_new / max(rs_old, 1e-12)
        for idx in range(len(p_w)):
            p_w[idx] = r_w[idx] + beta * p_w[idx]
        p_b = r_b + beta * p_b
        rs_old = rs_new
    return x_w, x_b


def influence_score(
    model: HashedTextLogReg,
    train_record: DocumentRecord,
    validation_records: Sequence[DocumentRecord],
    train_records: Sequence[DocumentRecord],
) -> float:
    train_grad_w, train_grad_b = model.average_gradient([train_record])
    val_grad_w, val_grad_b = model.average_gradient(validation_records)
    inverse_hvp_w, inverse_hvp_b = conjugate_gradient(
        lambda vec_w, vec_b: model.hessian_vector_product(train_records, vec_w, vec_b),
        train_grad_w,
        train_grad_b,
    )
    return -dot(val_grad_w, val_grad_b, inverse_hvp_w, inverse_hvp_b)


class ProxyEvaluator:
    def __init__(
        self,
        dim: int = 256,
        lr: float = 0.15,
        epochs: int = 12,
        l2: float = 1e-3,
        target_scale: float = 7_000_000_000.0,
        proxy_scale: float = 350_000_000.0,
        scaling_alpha: float = 0.28,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.target_scale = target_scale
        self.proxy_scale = proxy_scale
        self.scaling_alpha = scaling_alpha

    def build_model(self) -> HashedTextLogReg:
        return HashedTextLogReg(dim=self.dim, lr=self.lr, epochs=self.epochs, l2=self.l2)

    def evaluate_subset(self, train_records: Sequence[DocumentRecord], val_records: Sequence[DocumentRecord]) -> Dict[str, float]:
        model = self.build_model()
        model.fit(train_records)
        metrics = model.evaluate(val_records)
        metrics["scaled_value"] = self.extrapolate(metrics["value"])
        return metrics

    def extrapolate(self, proxy_gain: float) -> float:
        scale_ratio = self.proxy_scale / self.target_scale
        return proxy_gain * (scale_ratio ** self.scaling_alpha)

    def source_gains(
        self,
        train_records: Sequence[DocumentRecord],
        val_records: Sequence[DocumentRecord],
    ) -> Dict[str, float]:
        full_value = self.evaluate_subset(train_records, val_records)["scaled_value"]
        grouped: Dict[str, List[DocumentRecord]] = defaultdict(list)
        for record in train_records:
            grouped[record.source_id].append(record)
        gains: Dict[str, float] = {}
        for source_id in grouped:
            ablated = [record for record in train_records if record.source_id != source_id]
            ablated_value = self.evaluate_subset(ablated, val_records)["scaled_value"]
            gains[source_id] = full_value - ablated_value
        return gains

    def source_influences(
        self,
        train_records: Sequence[DocumentRecord],
        val_records: Sequence[DocumentRecord],
    ) -> Dict[str, float]:
        model = self.build_model()
        model.fit(train_records)
        grouped: Dict[str, List[DocumentRecord]] = defaultdict(list)
        for record in train_records:
            grouped[record.source_id].append(record)
        scores: Dict[str, float] = {}
        for source_id, records in grouped.items():
            if not records:
                scores[source_id] = 0.0
                continue
            per_doc = [
                influence_score(model, record, val_records, train_records)
                for record in records[: min(5, len(records))]
            ]
            scores[source_id] = sum(per_doc) / len(per_doc)
        return scores
