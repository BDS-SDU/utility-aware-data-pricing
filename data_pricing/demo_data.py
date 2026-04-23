from __future__ import annotations

from typing import Dict, List, Tuple


def build_demo_rows() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []

    sources = {
        "api_gold": {
            "label": 1,
            "texts": [
                "The handler validates the request payload, checks the auth token, and returns a typed response with status code and retry policy.",
                "API reference: create_session takes user_id, ttl_seconds, and scope list, then returns a session object with expiration metadata.",
                "The caching layer uses write through mode so repeated reads avoid stale state and reduce latency in the inference path.",
                "Code example: initialize the client, pass timeout and region, then inspect the structured error field before retrying.",
            ],
        },
        "reasoning_gold": {
            "label": 1,
            "texts": [
                "To debug the failing job, first isolate the malformed record, then compare schema evolution logs, and finally re-run the partition with corrected offsets.",
                "A correct migration plan requires backing up the table, applying the index change, validating row counts, and rolling forward only after checksum parity.",
                "The algorithm sorts candidate paths, prunes dominated branches, and then chooses the minimum cost route under the capacity constraint.",
                "Reasoning trace: observe the null pointer, inspect constructor ordering, infer an uninitialized dependency, and fix the boot sequence.",
            ],
        },
        "faq_mixed": {
            "label": 1,
            "texts": [
                "Question: how do I rotate service credentials. Answer: issue a new key, update environment variables, redeploy workers, and revoke the old key.",
                "This note explains deployment windows, maintenance contacts, and the expected rollback path if the canary metrics regress.",
                "A tutorial page mixes user guidance with some repetitive onboarding language, but still includes concrete command examples and troubleshooting steps.",
                "The document lists common failure modes, remediation owners, and escalation conditions for production incidents.",
            ],
        },
        "redundant_copy": {
            "label": 1,
            "texts": [
                "Click save to save your settings. Click save to save your settings. Click save to save your settings.",
                "The system is easy to use and easy to learn. The system is easy to use and easy to learn.",
                "Quick start quick start quick start, follow the quick start, use the quick start, repeat the same quick start.",
                "Configuration page configuration page configuration page with repeated wording and little technical novelty.",
            ],
        },
        "noise_dump": {
            "label": 0,
            "texts": [
                "lorem zxqv broken token stream ??? ??? repeated junk with no stable structure or recoverable technical meaning",
                "random paste #### 123 123 123 error nonsense table table table and inconsistent bracket { [ syntax mess",
                "chat fragments mixed with spam links and shallow repetition, offering no reliable supervision signal",
                "garbled note where fields are missing, delimiters drift, and semantics collapse into noise noise noise",
            ],
        },
    }

    for source_id, spec in sources.items():
        for idx, text in enumerate(spec["texts"], start=1):
            train_rows.append(
                {
                    "doc_id": f"{source_id}-train-{idx}",
                    "source_id": source_id,
                    "text": text,
                    "label": spec["label"],
                    "metadata": {"split": "train"},
                }
            )

    validation_examples = [
        ("val-1", "The endpoint returns structured JSON with retry_after and request_id fields for observability.", 1),
        ("val-2", "To recover the worker, inspect the queue lag, replay the failed batch, and verify downstream consistency.", 1),
        ("val-3", "API docs should explain parameters, return schema, and failure handling with precise examples.", 1),
        ("val-4", "Repeated filler text with no technical depth should not dominate the value estimate.", 0),
        ("val-5", "Corrupted snippets and malformed delimiters reduce training utility for downstream tasks.", 0),
        ("val-6", "A noisy dump without stable semantics adds little value to a documentation assistant.", 0),
    ]
    for doc_id, text, label in validation_examples:
        val_rows.append(
            {
                "doc_id": doc_id,
                "source_id": "validation",
                "text": text,
                "label": label,
                "metadata": {"split": "val", "eval_weight": 1.0 if label == 1 else 0.25},
            }
        )
    return train_rows, val_rows
