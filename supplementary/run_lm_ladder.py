"""Cross-scale leave-one-source-out (LOSO) experiment with real language models.

For every Pythia checkpoint (70m / 160m / 410m) and every domain corpus built
by build_data.py:

  1. fine-tune the pretrained model on the full domain corpus (all shards),
  2. for each source shard i, fine-tune a fresh copy on the corpus without
     shard i,
  3. evaluate each fine-tuned model as mean per-token NLL over the held-out
     validation documents (each document evaluated in isolation),
  4. define the source gain  G_i = NLL(D \\ D_i) - NLL(D)   (positive means
     the shard lowers validation NLL, i.e. helps the model).

This is the direct realisation of the "target training regime" that the
paper's Eq. (10) proxy gain is supposed to approximate (paper Section 2.2:
the proxy-vs-target stability assumption).

Usage (defaults reproduce the supplementary run):
    python supplementary/run_lm_ladder.py --models pythia-70m pythia-160m pythia-410m
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

SEQ_LEN = 512
EPOCHS = 4
LR = 1e-4
BATCH_SIZE = 8
WARMUP = 10
WEIGHT_DECAY = 0.01
SEED = 17

MODEL_PARAMS = {
    "EleutherAI/pythia-70m": 70_426_624,
    "EleutherAI/pythia-160m": 162_322_944,
    "EleutherAI/pythia-410m": 405_334_016,
}


@dataclass
class Doc:
    doc_id: str
    source_id: str
    text: str
    split: str


def load_domain(path: Path) -> Dict[str, List[Doc]]:
    train, val = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc = Doc(row["doc_id"], row["source_id"], row["text"], row["split"])
            (val if row["split"] == "val" else train).append(doc)
    return {"train": train, "val": val}


def tokenize_docs(tokenizer, docs: Sequence[Doc]) -> Dict[str, List[int]]:
    """Tokenise each doc independently and append EOS (paper corpora are
    self-contained documents)."""
    out: Dict[str, List[int]] = {}
    eos = tokenizer.eos_token_id
    for doc in docs:
        ids = tokenizer(doc.text, add_special_tokens=False)["input_ids"]
        out[doc.doc_id] = ids + [eos]
    return out


def make_chunks(token_lists: Sequence[List[int]], seq_len: int) -> List[List[int]]:
    """Concatenate documents into a single token stream, then cut fixed-length
    training chunks (standard continued-pretraining protocol)."""
    stream: List[int] = []
    for ids in token_lists:
        stream.extend(ids)
    chunks = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
    chunks = [c for c in chunks if len(c) >= 16]  # drop tiny tail fragment
    return chunks


def finetune(model, chunks: List[List[int]], epochs: int, lr: float, batch_size: int, seed: int, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(chunks) / batch_size))
    total_steps = steps_per_epoch * epochs
    generator = torch.Generator().manual_seed(seed)

    def lr_lambda(step: int) -> float:
        if step < WARMUP:
            return (step + 1) / max(1, WARMUP)
        progress = (step - WARMUP) / max(1, total_steps - WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    for _epoch in range(epochs):
        order = torch.randperm(len(chunks), generator=generator).tolist()
        for start in range(0, len(chunks), batch_size):
            batch_ids = [chunks[i] for i in order[start : start + batch_size]]
            maxlen = max(len(c) for c in batch_ids)
            input_ids = torch.full((len(batch_ids), maxlen), 0, dtype=torch.long)
            labels = torch.full((len(batch_ids), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            for bi, chunk in enumerate(batch_ids):
                input_ids[bi, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
                labels[bi, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
                attn[bi, : len(chunk)] = 1
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    return step


@torch.no_grad()
def eval_docs(model, tokenized: Dict[str, List[int]], seq_len: int, device) -> Dict[str, Dict[str, float]]:
    """Per-document mean NLL; docs longer than seq_len are split into chunks."""
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    for doc_id, ids in tokenized.items():
        chunks = [ids[i : i + seq_len] for i in range(0, len(ids), seq_len)]
        chunks = [c for c in chunks if len(c) >= 2]
        total_nll, total_tok = 0.0, 0
        for chunk in chunks:
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            labels = input_ids.clone()
            logits = model(input_ids=input_ids).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tok += shift_labels.numel()
        results[doc_id] = {
            "nll_sum": total_nll,
            "tokens": total_tok,
            "nll_per_token": total_nll / max(1, total_tok),
        }
    return results


def pooled_nll(per_doc: Dict[str, Dict[str, float]]) -> float:
    nll_sum = sum(v["nll_sum"] for v in per_doc.values())
    tokens = sum(v["tokens"] for v in per_doc.values())
    return nll_sum / max(1, tokens)


def run_model_on_domain(model_name: str, domain: str, data: Dict[str, List[Doc]], args) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.config.use_cache = False
    torch.backends.cuda.matmul.allow_tf32 = True

    train_docs = data["train"]
    val_docs = data["val"]
    val_tokens = tokenize_docs(tokenizer, val_docs)
    train_by_source: Dict[str, List[Doc]] = {}
    for doc in train_docs:
        train_by_source.setdefault(doc.source_id, []).append(doc)
    sources = sorted(train_by_source.keys())

    runs: Dict[str, object] = {}
    eos = tokenizer.eos_token_id
    train_tokens_by_doc = {
        d.doc_id: tokenizer(d.text, add_special_tokens=False)["input_ids"] + [eos] for d in train_docs
    }

    def train_and_eval(docs: List[Doc], tag: str, seed_offset: int) -> float:
        torch.manual_seed(args.seed + seed_offset)
        model.load_state_dict(base_state)  # fresh pretrained init every run
        chunks = make_chunks([train_tokens_by_doc[d.doc_id] for d in docs], args.seq_len)
        t0 = time.time()
        finetune(model, chunks, args.epochs, args.lr, args.batch_size, args.seed + seed_offset, device)
        per_doc = eval_docs(model, val_tokens, args.seq_len, device)
        pooled = pooled_nll(per_doc)
        runs[tag] = {
            "n_train_docs": len(docs),
            "n_chunks": len(chunks),
            "pooled_nll": pooled,
            "train_seconds": round(time.time() - t0, 1),
            "per_doc": {k: round(v["nll_per_token"], 6) for k, v in per_doc.items()},
            "per_doc_tokens": {k: v["tokens"] for k, v in per_doc.items()},
        }
        print(f"  [{domain}/{model_name}] {tag}: nll={pooled:.6f} docs={len(docs)} chunks={len(chunks)} ({runs[tag]['train_seconds']}s)", flush=True)
        return pooled

    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    full_nll = train_and_eval(train_docs, "full", 0)
    gains: Dict[str, Dict[str, float]] = {}
    for si, source in enumerate(sources, start=1):
        ablated = [d for d in train_docs if d.source_id != source]
        ablated_nll = train_and_eval(ablated, f"loso_{source}", si)
        gains[source] = {
            "gain": ablated_nll - full_nll,  # positive => shard helps
            "full_nll": full_nll,
            "ablated_nll": ablated_nll,
        }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {
        "model": model_name.split("/")[-1],
        "model_hf_id": model_name,
        "params": MODEL_PARAMS.get(model_name),
        "domain": domain,
        "sources": sources,
        "full_nll": full_nll,
        "gains": gains,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"])
    parser.add_argument("--domains", nargs="*", default=None, help="domain jsonl stems, default all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_files = sorted(args.data_dir.glob("*.jsonl")) if not args.domains else [args.data_dir / f"{d}.jsonl" for d in args.domains]
    domains = {path.stem: load_domain(path) for path in domain_files}

    for model_name in args.models:
        model_key = model_name.split("/")[-1]
        results = []
        for domain, data in domains.items():
            payload = run_model_on_domain(model_name, domain, data, args)
            results.append(payload)
            detail_path = args.out_dir / f"lm_detail_{model_key}_{domain}.json"
            detail_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # flat CSV of gains
        csv_path = args.out_dir / f"lm_gains_{model_key}.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("model,params,domain,source,gain,full_nll,ablated_nll\n")
            for payload in results:
                for source, g in payload["gains"].items():
                    handle.write(
                        f"{payload['model']},{payload['params']},{payload['domain']},{source},"
                        f"{g['gain']:.8f},{g['full_nll']:.8f},{g['ablated_nll']:.8f}\n"
                    )
        print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
