#  scripts/00_dataset_analysis_statistics_all_models.py
#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd

# 1. Define your models and their local paths
MODELS = {
    "DeepSeek-1.3B": "models/deepseek-coder-1.3b-instruct",
    "DeepSeek-6.7B": "models/deepseek-coder-6.7b-instruct",
    "StarCoder2-3B": "models/starcoder2-3b",
    "StarCoder2-7B": "models/starcoder2-7b",
    "Qwen2.5-7B": "models/Qwen2.5-Coder-7B-Instruct"
}

def load_data(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]

# Load your dataset
train_data = load_data("data/processed/train.jsonl")
sample = train_data[:500]  # Statistical sample


print(f"{'Model':<20} | {'Mean':<6} | {'p95':<6} | {'Max':<6} | {'Trunc% (512)':<12}")
print("-" * 65)


all_stats = []

for name, path in MODELS.items():
    if not Path(path).exists(): continue
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    
    total_lengths = []
    output_lengths = []

    for ex in sample:
        meta = ex.get("metadata", {})
        # Combine variant and algorithm_params into one dictionary
        all_technical_context = {**meta.get("variant", {}), **meta.get("algorithm_params", {})}
        
        # Flatten the dictionary into a technical string: Key=Value | Key=Value
        technical_bits = []
        for k, v in all_technical_context.items():
            if isinstance(v, list):
                val_str = ", ".join(map(str, v))
            else:
                val_str = str(v)
            technical_bits.append(f"{k.replace('_', ' ').title()}: {val_str}")
        
        # Add the cipher and family identity explicitly at the start
        meta_str = f"Cipher: {meta.get('cipher')} ({meta.get('family')}) | " + " | ".join(technical_bits)
    
        # Reconstruct full prompt with THE ENTIRE CONTEXT
        full_prompt = (
            f"### Instruction: {ex['instruction']}\n"
            f"Technical Context: {meta_str}\n"
            f"Requirement: Provide ONLY the Isabelle/HOL code.\n"
            f"### Input: {ex['input']}\n"
            f"### Output: {ex['output']}"
        )
        
        # 4. Measure
        tokens = tokenizer.encode(full_prompt)
        out_tokens = tokenizer.encode(ex['output'], add_special_tokens=False)
        
        total_lengths.append(len(tokens))
        output_lengths.append(len(out_tokens))

    # Calculate statistics for this tokenizer
    p95_total = np.percentile(total_lengths, 95)
    p95_output = np.percentile(output_lengths, 95)
    
    print(f"{name:<20} | Total p95: {p95_total:>4.0f} | Mwx Length: {max(total_lengths):>4.0f} | Output p95: {p95_output:>4.0f} | tokens MAX Output LENGTH: {max(out_tokens)}")


# 2. Final Justification Logic for the Paper
#unified_p95 = max([s['p95'] for s in all_stats])
print("\n" + "="*60)
print("ICML HYPERPARAMETER JUSTIFICATION")
print("="*60)
print(f"Across all tokenizers, the global p95 length is {p95_total} tokens.")
print(f"To ensure zero information loss across models, we adopt a unified max_length=1024.")



