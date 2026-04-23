from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, List, Sequence

from .data_types import DocumentRecord, DocumentScore, SourceScore, ValuationReport
from .quality import NGramLanguageModel, score_document, simple_tokenize
from .shapley import mean_confidence_interval, monte_carlo_source_shapley
from .valuation import ProxyEvaluator
from .verification import ProofOfTrainingLedger


def minmax_scale(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if lower == upper:
        return {key: 0.5 for key in values}
    return {key: (value - lower) / (upper - lower) for key, value in values.items()}


class DynamicDataValuationPipeline:
    def __init__(
        self,
        dqs_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        ensemble_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.2, 0.2),
        ngram_order: int = 3,
    ) -> None:
        self.dqs_weights = dqs_weights
        self.ensemble_weights = ensemble_weights
        self.ngram_order = ngram_order
        self.proxy_evaluator = ProxyEvaluator()

    def _reference_model(self, records: Sequence[DocumentRecord]) -> NGramLanguageModel:
        model = NGramLanguageModel(order=self.ngram_order)
        model.fit(simple_tokenize(record.text) for record in records)
        return model

    def _score_documents(
        self,
        records: Sequence[DocumentRecord],
        reference_model: NGramLanguageModel,
    ) -> List[DocumentScore]:
        return [score_document(record, reference_model, self.dqs_weights) for record in records]

    def run(
        self,
        train_records: Sequence[DocumentRecord],
        val_records: Sequence[DocumentRecord],
        shapley_iterations: int = 64,
    ) -> ValuationReport:
        reference_model = self._reference_model(train_records)
        document_scores = self._score_documents(train_records, reference_model)
        grouped_scores: Dict[str, List[DocumentScore]] = defaultdict(list)
        grouped_records: Dict[str, List[DocumentRecord]] = defaultdict(list)
        for doc_score in document_scores:
            grouped_scores[doc_score.source_id].append(doc_score)
        for record in train_records:
            grouped_records[record.source_id].append(record)

        proxy_gains = self.proxy_evaluator.source_gains(train_records, val_records)
        influences = self.proxy_evaluator.source_influences(train_records, val_records)
        shapley_values = monte_carlo_source_shapley(
            train_records=train_records,
            val_records=val_records,
            evaluator=self.proxy_evaluator,
            iterations=shapley_iterations,
        )

        dqs_means = {key: mean(item.dqs for item in values) for key, values in grouped_scores.items()}
        dqs_scaled = minmax_scale(dqs_means)
        proxy_scaled = minmax_scale(proxy_gains)
        influence_scaled = minmax_scale(influences)
        shapley_scaled = minmax_scale(shapley_values)

        model = self.proxy_evaluator.build_model()
        ledger = ProofOfTrainingLedger([record.doc_id for record in train_records])
        ledger.start(model.weights, model.bias)
        model.fit(train_records)
        ledger.append_step(
            step=1,
            batch_doc_ids=[record.doc_id for record in train_records[: min(8, len(train_records))]],
            metrics=model.evaluate(val_records),
            weights=model.weights,
            bias=model.bias,
        )

        source_scores: List[SourceScore] = []
        for source_id, docs in grouped_scores.items():
            components = [
                dqs_scaled.get(source_id, 0.0),
                proxy_scaled.get(source_id, 0.0),
                influence_scaled.get(source_id, 0.0),
                shapley_scaled.get(source_id, 0.0),
            ]
            unified_score = sum(weight * value for weight, value in zip(self.ensemble_weights, components))
            ci_low, ci_high = mean_confidence_interval(
                [
                    proxy_gains.get(source_id, 0.0),
                    influences.get(source_id, 0.0),
                    shapley_values.get(source_id, 0.0),
                ]
            )
            source_scores.append(
                SourceScore(
                    source_id=source_id,
                    document_count=len(grouped_records[source_id]),
                    token_count=sum(doc.token_count for doc in docs),
                    mean_dqs=dqs_means[source_id],
                    proxy_gain=proxy_gains.get(source_id, 0.0),
                    influence_score=influences.get(source_id, 0.0),
                    shapley_value=shapley_values.get(source_id, 0.0),
                    unified_score=unified_score,
                    confidence_low=ci_low,
                    confidence_high=ci_high,
                )
            )

        source_scores.sort(key=lambda item: item.unified_score, reverse=True)
        summary = {
            "num_train_records": len(train_records),
            "num_val_records": len(val_records),
            "num_sources": len(grouped_records),
            "top_source": source_scores[0].source_id if source_scores else None,
            "mean_document_dqs": mean(item.dqs for item in document_scores) if document_scores else 0.0,
        }
        config = {
            "dqs_weights": self.dqs_weights,
            "ensemble_weights": self.ensemble_weights,
            "ngram_order": self.ngram_order,
            "shapley_iterations": shapley_iterations,
        }
        return ValuationReport(
            document_scores=document_scores,
            source_scores=source_scores,
            ledger=ledger.fingerprint(),
            config=config,
            summary=summary,
        )
