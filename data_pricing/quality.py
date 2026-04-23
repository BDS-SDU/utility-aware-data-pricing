from __future__ import annotations

import math
import re
import hashlib
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_types import DocumentRecord, DocumentScore

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"[.!?\n]+")


def simple_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def stable_hash_index(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % dim


class NGramLanguageModel:
    """Additive-smoothed n-gram model used as a lightweight reference LM."""

    def __init__(self, order: int = 3, alpha: float = 0.5) -> None:
        self.order = max(1, order)
        self.alpha = alpha
        self.context_counts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        for tokens in sequences:
            padded = ["<bos>"] * (self.order - 1) + list(tokens)
            self.vocabulary.update(tokens)
            for idx in range(self.order - 1, len(padded)):
                context = tuple(padded[idx - self.order + 1 : idx])
                token = padded[idx]
                self.context_counts[context][token] += 1

    def conditional_probability(self, context: Sequence[str], token: str) -> float:
        ctx = tuple((["<bos>"] * (self.order - 1) + list(context))[-(self.order - 1) :])
        counts = self.context_counts.get(ctx)
        vocab_size = max(1, len(self.vocabulary))
        if not counts:
            return 1.0 / vocab_size
        total = sum(counts.values())
        return (counts[token] + self.alpha) / (total + self.alpha * vocab_size)

    def mean_surprisal_bits(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        surprises: List[float] = []
        history: List[str] = []
        for token in tokens:
            prob = self.conditional_probability(history, token)
            surprises.append(-math.log(prob, 2))
            history.append(token)
        return sum(surprises) / len(surprises)


def normalize_info_density(bits_per_token: float, cap: float = 8.0) -> float:
    return max(0.0, min(bits_per_token / cap, 1.0))


def _balanced_pairs(text: str, left: str, right: str) -> bool:
    balance = 0
    for char in text:
        if char == left:
            balance += 1
        elif char == right:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0


def syntactic_coherence(text: str) -> float:
    if not text.strip():
        return 0.0
    checks = []
    checks.append(1.0 if _balanced_pairs(text, "(", ")") else 0.0)
    checks.append(1.0 if _balanced_pairs(text, "[", "]") else 0.0)
    checks.append(1.0 if _balanced_pairs(text, "{", "}") else 0.0)
    quote_count = text.count('"')
    checks.append(1.0 if quote_count % 2 == 0 else 0.0)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    long_lines = sum(1 for line in lines if len(line) > 240)
    checks.append(max(0.0, 1.0 - long_lines / max(1, len(lines))))
    alpha_ratio = sum(ch.isalnum() or ch.isspace() for ch in text) / max(1, len(text))
    checks.append(min(max(alpha_ratio, 0.0), 1.0))
    tokens = simple_tokenize(text)
    if tokens:
        alpha_tokens = sum(token.isalpha() for token in tokens) / len(tokens)
        punct_tokens = sum(len(token) == 1 and not token.isalnum() for token in tokens) / len(tokens)
        malformed = 0
        for token in tokens:
            if token.isalnum():
                if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
                    malformed += 1
                elif token.isalpha() and len(token) >= 6 and not any(vowel in token for vowel in "aeiou"):
                    malformed += 1
            elif len(token) > 1:
                malformed += 1
        checks.append(alpha_tokens)
        checks.append(max(0.0, 1.0 - punct_tokens * 2.0))
        checks.append(max(0.0, 1.0 - malformed / len(tokens)))
    return sum(checks) / len(checks)


def split_segments(text: str) -> List[List[str]]:
    segments = []
    for chunk in SENTENCE_SPLIT.split(text):
        tokens = simple_tokenize(chunk)
        if tokens:
            segments.append(tokens)
    return segments[:16]


def _hashed_vector(tokens: Sequence[str], dim: int = 128) -> List[float]:
    vector = [0.0] * dim
    for token in tokens:
        vector[stable_hash_index(token, dim)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def semantic_richness(text: str) -> float:
    segments = split_segments(text)
    if len(segments) <= 1:
        return 0.5
    vectors = [_hashed_vector(segment) for segment in segments]
    similarities: List[float] = []
    for idx, left in enumerate(vectors):
        for right in vectors[idx + 1 :]:
            similarities.append(cosine_similarity(left, right))
    if not similarities:
        return 0.5
    redundancy = sum(similarities) / len(similarities)
    diversity = max(0.0, min(1.0, 1.0 - redundancy))
    token_pool = [token for segment in segments for token in segment]
    if not token_pool:
        return diversity
    alpha_tokens = sum(token.isalpha() for token in token_pool) / len(token_pool)
    repeated_tokens = 1.0 - (len(set(token_pool)) / len(token_pool))
    lexical_factor = max(0.0, min(1.0, 0.7 * alpha_tokens + 0.3 * (1.0 - repeated_tokens)))
    return diversity * lexical_factor


def score_document(
    record: DocumentRecord,
    reference_model: NGramLanguageModel,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> DocumentScore:
    tokens = simple_tokenize(record.text)
    info_bits = reference_model.mean_surprisal_bits(tokens)
    info_norm = normalize_info_density(info_bits)
    syn = syntactic_coherence(record.text)
    sem = semantic_richness(record.text)
    dqs = weights[0] * info_norm + weights[1] * syn + weights[2] * sem
    return DocumentScore(
        doc_id=record.doc_id,
        source_id=record.source_id,
        token_count=len(tokens),
        info_density_bits=info_bits,
        normalized_info_density=info_norm,
        syntactic_coherence=syn,
        semantic_richness=sem,
        dqs=dqs,
    )
