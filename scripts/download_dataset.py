from datasets import load_dataset
import json

def dump(ds, path, n=300):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= n:
                break
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

dump(load_dataset("tatsu-lab/alpaca", split="train"), "data/alpaca_instruction.jsonl")
dump(load_dataset("openai/gsm8k", "main", split="train"), "data/gsm8k_math.jsonl")
dump(load_dataset("google/code_x_glue_ct_code_to_text", "python", split="train"), "data/codexglue_python.jsonl")
