from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .data_types import DocumentRecord
from .valuation import ProxyEvaluator


def monte_carlo_source_shapley(
    train_records: Sequence[DocumentRecord],
    val_records: Sequence[DocumentRecord],
    evaluator: ProxyEvaluator,
    iterations: int = 64,
    seed: int = 13,
) -> Dict[str, float]:
    grouped: Dict[str, List[DocumentRecord]] = defaultdict(list)
    for record in train_records:
        grouped[record.source_id].append(record)
    sources = list(grouped.keys())
    shapley = {source_id: 0.0 for source_id in sources}
    cache: Dict[Tuple[str, ...], float] = {tuple(): 0.0}
    randomizer = random.Random(seed)

    def value_of(source_subset: Sequence[str]) -> float:
        key = tuple(sorted(source_subset))
        if key not in cache:
            subset_records: List[DocumentRecord] = []
            for source_id in key:
                subset_records.extend(grouped[source_id])
            cache[key] = evaluator.evaluate_subset(subset_records, val_records)["scaled_value"]
        return cache[key]

    for _ in range(iterations):
        permutation = list(sources)
        randomizer.shuffle(permutation)
        prefix: List[str] = []
        prefix_value = value_of(prefix)
        for source_id in permutation:
            next_prefix = prefix + [source_id]
            next_value = value_of(next_prefix)
            shapley[source_id] += (next_value - prefix_value) / iterations
            prefix = next_prefix
            prefix_value = next_value
    return shapley


def mean_confidence_interval(values: Sequence[float], z_value: float = 1.96) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return mean - z_value * stderr, mean + z_value * stderr
