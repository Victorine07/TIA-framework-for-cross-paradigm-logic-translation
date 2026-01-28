#!/usr/bin/env python3
"""
OPTIMIZED Option 4 Pipeline 
1. Label masking - loss only on Isabelle output ✓
2. Higher LoRA rank for 1.3B (64) ✓
3. No repetition penalty for evaluation ✓  
4. Model-specific LoRA optimizations ✓
5. Reduced trainable params for 7B models ✓
6. Early stopping with larger patience ✓
7. Complete debugging and analysis ✓
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import re
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PATH SETUP
# ============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
#sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from evaluation.metrics_fixed import FixedCryptographicMetrics
    metrics = FixedCryptographicMetrics()
    print("✓ Using FixedCryptographicMetrics")
except ImportError:
    metrics = None
    print("✗ Metrics not available, will compute basic scores")
        
# ============================================================================
# DEBUG UTILITIES
# ============================================================================
def create_cross_validation_folds(data: List[Dict], n_folds: int = 3):  # Changed to 3
    """Create cross-validation folds for supplementary analysis."""
    np.random.seed(42)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    folds = []
    fold_size = len(data) // n_folds
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds - 1 else len(data)
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_fold = [data[idx] for idx in train_indices]
        val_fold = [data[idx] for idx in val_indices]
        
        folds.append((train_fold, val_fold))
    
    return folds
    
class TrainingDebugger:
    """Debug training issues with comprehensive checks."""
    
    @staticmethod
    def check_label_alignment(tokenizer, tokenized_example):
        """Check if labels are correctly aligned with input."""
        input_ids = tokenized_example["input_ids"]
        labels = tokenized_example["labels"] if "labels" in tokenized_example else None
        
        print("\n=== LABEL ALIGNMENT DEBUG ===")
        print(f"Input IDs length: {len(input_ids)}")
        
        if labels is not None:
            print(f"Labels length: {len(labels)}")
            non_ignore = sum(1 for l in labels if l != -100)
            print(f"Non-ignored labels: {non_ignore}")
            print(f"Ignored labels: {len(labels) - non_ignore}")
            
            # Show where labels start
            for i, (inp, lab) in enumerate(zip(input_ids, labels)):
                if lab != -100:
                    print(f"First non-ignore at position {i}:")
                    print(f"  Input token: {inp} -> '{tokenizer.decode([inp])}'")
                    print(f"  Label token: {lab} -> '{tokenizer.decode([lab])}'")
                    break
        else:
            print("No labels found in example")
        
        # Decode first 20 tokens
        print(f"\nFirst 20 tokens:")
        for i in range(min(20, len(input_ids))):
            token = input_ids[i]
            print(f"  {i:3d}: {token:6d} -> '{tokenizer.decode([token])}'")
        
        return True
    
    @staticmethod
    def analyze_generated_text(generated: str, reference: str, example_idx: int = 0):
        """Analyze generated text vs reference."""
        print(f"\n=== GENERATION ANALYSIS (Example {example_idx}) ===")
        print(f"Generated length: {len(generated)} chars")
        print(f"Reference length: {len(reference)} chars")
        
        # Check for common patterns
        isabelle_keywords = ["definition", "fun", "lemma", "where", "::", "="]
        generated_has = [kw for kw in isabelle_keywords if kw in generated]
        reference_has = [kw for kw in isabelle_keywords if kw in reference]
        
        print(f"Generated has keywords: {generated_has}")
        print(f"Reference has keywords: {reference_has}")
        
        # Show first few lines
        print("\nGenerated (first 5 lines):")
        for i, line in enumerate(generated.split('\n')[:5]):
            print(f"  {i}: {line}")
        
        print("\nReference (first 5 lines):")
        for i, line in enumerate(reference.split('\n')[:5]):
            print(f"  {i}: {line}")

# ============================================================================
# OPTIMIZED EVALUATION WITH ALL FIXES
# ============================================================================
def optimize_model_for_evaluation(model, use_4bit: bool = True):
    """Optimize model for faster inference with all fixes."""
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
        # For Qwen models
        if hasattr(model.config, 'model_type') and 'qwen' in model.config.model_type.lower():
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
        
        torch.cuda.empty_cache()
    
    return model

def batch_generate_with_cache(model, tokenizer, prompts: List[str], batch_size: int = 2, 
                             max_new_tokens: int = 512, use_cache: bool = True):
    """Batch generation with NO repetition penalty (CRITICAL FIX)."""
    all_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Deterministic
                temperature=0.0,          # Greedy decoding
                top_p=None,               # No top-p sampling
                num_beams=1,              # Single beam (greedy)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,   # CRITICAL: No penalty for formal code
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
        
        for j, output_ids in enumerate(outputs.sequences):
            prompt_len = len(inputs['input_ids'][j])
            generated_ids = output_ids[prompt_len:]
            
            # Skip if nothing generated
            if len(generated_ids) == 0:
                all_outputs.append("")
                continue
            
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated = extract_isabelle_code(generated)
            all_outputs.append(generated)
        
        # Clear cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_outputs

def extract_isabelle_code(text: str) -> str:
    """Extract Isabelle/HOL code from generated text."""
    # Remove markdown code blocks if present
    text = re.sub(r'```\w*\n', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Common Isabelle patterns
    patterns = [
        r'(definition\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
        r'(fun\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
        r'(lemma\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0][0].strip()
            
            # Clean up: remove trailing incomplete lines
            lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped:
                    # Check if line looks complete
                    if stripped.startswith(('definition', 'fun', 'lemma')):
                        if not (stripped.endswith('where') or '::' in stripped):
                            # Incomplete definition start, skip
                            continue
                    lines.append(line)
            
            if lines:
                return '\n'.join(lines)
    
    # Fallback: look for any Isabelle-like code
    lines = []
    in_isabelle_block = False
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Start of Isabelle code
        if stripped.startswith(('definition', 'fun', 'lemma')):
            in_isabelle_block = True
            lines.append(line)
        elif in_isabelle_block:
            # Continue until we hit something that doesn't look like Isabelle
            if re.match(r'^\s*(?:definition|fun|lemma|where|::|=|"|\||end)', stripped):
                lines.append(line)
            else:
                break
        elif '::' in stripped or 'where' in stripped:
            # Might be continuation
            lines.append(line)
    
    if lines:
        return '\n'.join(lines)
    
    # Last resort: clean and return
    return clean_code(text)

def clean_code(code_str: str) -> str:
    """Clean Isabelle/HOL code."""
    import re
    
    # Remove comments
    code_str = re.sub(r'\(\*.*?\*\)', '', code_str, flags=re.DOTALL)
    code_str = re.sub(r'#.*', '', code_str)
    
    # Remove extra whitespace
    lines = [line.rstrip() for line in code_str.splitlines() if line.strip()]
    
    # Remove empty lines at beginning and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    
    return '\n'.join(lines)

# ============================================================================
# DATA-DRIVEN CONFIGURATION WITH MODEL-SPECIFIC OPTIMIZATIONS
# ============================================================================
def get_data_driven_config(model_size: str, model_type: str = "unknown") -> dict:
    """
    Unified parameters with model-specific optimizations.
    """
    # Base configurations
    base_configs = {
        "1.3b": {
            "max_length": 1024,
            "num_epochs": 4,  
            "batch_size": 4,
            "gradient_accumulation": 4,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_4bit": True,
        },
        "3b": {
            "max_length": 1024,
            "num_epochs": 4,
            "batch_size": 2,
            "gradient_accumulation": 8,
            "learning_rate": 3e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_4bit": True,
        },
        "7b": {
            "max_length": 1024,
            "num_epochs": 4,
            "batch_size": 1,  # MUST be 1 for 7B on 16GB GPU
            "gradient_accumulation": 16,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_4bit": True,
        }
    }
    
    ms = model_size.lower().strip()
    
    # Detect model size
    if any(pattern in ms for pattern in ["1.3", "1_3", "1-3"]):
        config = base_configs["1.3b"].copy()
        config["num_epochs"] = 4  # Reduced from 6
    elif "3b" in ms and "7" not in ms and "13" not in ms:
        config = base_configs["3b"].copy()
        config["num_epochs"] = 4  # Reduced from 6
    else:
        config = base_configs["7b"].copy()
        config["num_epochs"] = 4  # Reduced from 6
    
    # Model-specific LoRA configurations 
    if "1.3" in ms:
        # DeepSeek 1.3B needs HIGHER rank for capacity
        config.update({
            "lora_rank": 64,     # Increased from 32
            "lora_alpha": 128,   # 2× rank
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
        })
        print("  LoRA config: rank=64, alpha=128 (high capacity for 1.3B)")
    
    elif "7b" in ms or "6.7b" in ms:
        # 7B models: FEWER parameters for efficiency
        if "qwen" in model_type.lower():
            # Qwen 7B: Minimal LoRA
            config.update({
                "lora_rank": 32,      # Very low rank
                "lora_alpha": 64,    # 2× rank
                "target_modules": ["q_proj", "v_proj"],  # Only Q, V
                "lora_dropout": 0.1,  # Lower dropout
            })
            print("  LoRA config: rank=8, alpha=16 (minimal for Qwen 7B)")
        elif "starcoder" in model_type.lower():
            # StarCoder2 7B: Balanced
            config.update({
                "lora_rank": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
            })
            print("  LoRA config: rank=16, alpha=32 (balanced for StarCoder2)")
        else:
            # DeepSeek 6.7B/7B: Conservative
            config.update({
                "lora_rank": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
            })
            print("  LoRA config: rank=12, alpha=24 (conservative for 7B)")
    
    else:  # 3B models
        config.update({
            "lora_rank": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj", "k_proj"],
            "lora_dropout": 0.1,
        })
        print("  LoRA config: rank=32, alpha=64 (standard for 3B)")
    
    return config

# ============================================================================
# METADATA HANDLER
# ============================================================================
class EnhancedMetadataHandler:
    """Enhanced metadata handler with all strategies."""
    
    def __init__(self, strategy: str = "none"):
        self.strategy = strategy
        print(f"[Metadata] Strategy: {strategy}")
    
    def enrich(self, instruction: str, metadata: Dict) -> str:
        constraint = "\nRequirement: Provide ONLY the Isabelle/HOL code. No explanations."
        
        if self.strategy == "none" or not metadata:
            return f"### Task: {instruction}{constraint}"
        
        # Dispatch to strategy
        if self.strategy == "full":
            return self._enrich_full(instruction, metadata, constraint)
        elif self.strategy == "structured":
            return self._enrich_structured(instruction, metadata, constraint)
        elif self.strategy == "algorithmic":
            return self._enrich_algorithmic(instruction, metadata, constraint)
        elif self.strategy == "all_json":  # ADD THIS
            return self._enrich_all_json(instruction, metadata, constraint)
        else:
            # Unknown strategy, fallback to none
            return f"### Task: {instruction}{constraint}"
    
    def _enrich_full(self, instruction: str, metadata: Dict, constraint: str) -> str:
        tech_context = {**metadata.get("variant", {}), **metadata.get("algorithm_params", {})}
        technical_bits = []
        cipher = metadata.get('cipher', 'Unknown')
        family = metadata.get('family', 'Unknown')
        
        for k, v in tech_context.items():
            k_clean = k.replace('_', ' ').title()
            val_str = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            technical_bits.append(f"{k_clean}: {val_str}")
        
        spec_sheet = f"Cipher: {cipher} ({family}) | " + " | ".join(technical_bits)
        return f"### Technical Context: {spec_sheet}\n\n### Task: {instruction}{constraint}"
        
    def _enrich_structured(self, instruction: str, metadata: Dict, constraint: str) -> str:
        """
        REVISED Structured Metadata: Multi-Paradigm Parametric Anchoring.
        Handles ARX, Feistel, and SPN specifically using Tier-based parameters.
        """
        params = metadata.get("algorithm_params", {})
        variant = metadata.get("variant", {})
        cipher_name = metadata.get("cipher", "Unknown")
        family = metadata.get("family", "Unknown")
        
        if not params and not variant:
            return f"### Task: {instruction}{constraint}"
        
        # 1. CORE ARCHITECTURAL ANCHORS (Always include)
        priority_params = [
            ("Cipher", f"{cipher_name} ({family})"),
            ("Block Size", variant.get("block_size", "N/A")),
            ("Word Size", params.get("word_size", variant.get("word_size", "N/A")))
        ]
        
        # 2. FAMILY-SPECIFIC LOGIC ANCHORS
        # ARX/Feistel: Focus on Rotations and Sequence
        if family in ["ARX", "Feistel"]:
            for key in ["alpha_rotation", "beta_rotation", "rotation_constants"]:
                if key in params:
                    priority_params.append((key.replace("_", " ").title(), params[key]))
            if "arx_order" in params:
                priority_params.append(("Operation Sequence", params["arx_order"]))
            if "f_function" in params:
                priority_params.append(("F-Function Logic", params["f_function"]))
    
        # SPN: Focus on S-Boxes and Nibbles
        elif family == "SPN":
            if "sbox_size" in params:
                priority_params.append(("S-Box Size", f"{params['sbox_size']}-bit"))
            if "sbox_count" in params:
                priority_params.append(("S-Box Count", params["sbox_count"]))
            if "permutation_type" in params:
                priority_params.append(("Permutation", params["permutation_type"]))
            if "nibbles_per_block" in variant:
                priority_params.append(("Nibble Count", variant["nibbles_per_block"]))
    
        # 3. STRUCTURAL ANCHORS (Iterations and Steps)
        if "rounds" in variant:
            priority_params.append(("Total Rounds", variant["rounds"]))
        elif "steps" in variant:
            priority_params.append(("Step Structure", f"{variant['steps']} steps × {variant.get('rounds_per_step', '')} rounds"))
    
        # 4. COMPONENT-SPECIFIC CONTEXT
        comp_type = metadata.get("component_type", "")
        if comp_type:
            priority_params.append(("Component Tier", comp_type))
    
        # Build structured list (Clean and Scannable)
        param_lines = []
        for key, value in priority_params:
            if isinstance(value, list):
                val_str = f"[{', '.join(map(str, value[:4]))}" + ("...]" if len(value) > 4 else "]")
            else:
                val_str = str(value)
            param_lines.append(f"- {key}: {val_str}")
        
        param_block = "\n".join(param_lines)
        return f"### Cryptographic Context:\n{param_block}\n\n### Task: {instruction}{constraint}"
    
         
    def _enrich_algorithmic(self, instruction: str, metadata: Dict, constraint: str) -> str:
        params = metadata.get("algorithm_params", {})
        
        core_info = []
        
        # Type information
        if "word_size" in params:
            core_info.append(f"{params['word_size']}b")
        
        # Operation pattern
        if "arx_order" in params:
            # Simplify ARX order
            order = params["arx_order"]
            if "→" in order:
                core_info.append(order.split("→")[0].strip())  # Just first operation
            else:
                core_info.append("ARX")
        elif "f_function" in params:
            f_func = str(params['f_function'])
            # Extract key operations
            if "<<" in f_func:
                core_info.append("rotate")
            if "&" in f_func:
                core_info.append("and")
            if "^" in f_func:
                core_info.append("xor")
        
        # Keep it very concise
        if core_info:
            spec_line = " + ".join(core_info[:3])  # Max 3 items
            return f"### Algorithm: {spec_line}\n\n### Task: {instruction}{constraint}"
        
        return f"### Task: {instruction}{constraint}"

    def _enrich_all_json(self, instruction: str, metadata: Dict, constraint: str) -> str:
        """FLATTENED ENTIRE JSON METADATA - NEW STRATEGY."""
        def flatten_dict(d, parent_key='', sep='.'):
            """Recursively flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Convert list to string
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten entire metadata
        flat_metadata = flatten_dict(metadata)
        
        # Build comprehensive spec sheet
        spec_lines = []
        for key, value in flat_metadata.items():
            # Clean up key names
            clean_key = key.replace('_', ' ').replace('.', ' → ').title()
            spec_lines.append(f"- {clean_key}: {value}")
        
        spec_block = "\n".join(spec_lines)
        return f"### Complete Metadata Specification:\n{spec_block}\n\n### Task: {instruction}{constraint}"
        

# ============================================================================
# DATA PROCESSING WITH PROPER LABEL MASKING
# ============================================================================
def load_all_training_data() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Load all training data for Option 4."""
    data_dir = Path("data/processed")
    unseen_dir = Path("data/unseen")
    
    # Load main dataset
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")
    
    all_data = train_data + val_data + test_data
    
    # Load unseen datasets
    lea_data = load_jsonl(unseen_dir / "lea_dataset.jsonl")
    rectangle_data = load_jsonl(unseen_dir / "rectangle_dataset.jsonl")
    xtea_data = load_jsonl(unseen_dir / "xtea_dataset.jsonl")
    
    # Create stratified subset for sanity check
    #training_subset = create_training_subset(all_data, n_samples=90)
    training_subset = test_data
    
    print(f"\nDATA SUMMARY:")
    print(f"  All training: {len(all_data)} examples")
    print(f"  LEA (validation): {len(lea_data)} examples")
    print(f"  RECTANGLE (test): {len(rectangle_data)} examples")
    print(f"  XTEA (test): {len(xtea_data)} examples")
    print(f"  Training subset: {len(training_subset)} examples")
    
    return all_data, lea_data, rectangle_data, xtea_data, training_subset

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def create_training_subset(data: List[Dict], n_samples: int = 30) -> List[Dict]:
    """Create stratified training subset."""
    import numpy as np
    
    # Group by family
    families = {}
    for example in data:
        family = example.get("metadata", {}).get("family", "unknown")
        families.setdefault(family, []).append(example)
    
    subset = []
    samples_per_family = max(1, n_samples // len(families))
    
    for family, examples in families.items():
        if len(examples) <= samples_per_family:
            subset.extend(examples)
        else:
            indices = np.random.choice(len(examples), samples_per_family, replace=False)
            subset.extend([examples[i] for i in indices])
    
    # Fill if needed
    if len(subset) < n_samples:
        remaining = [ex for ex in data if ex not in subset]
        if remaining:
            additional = n_samples - len(subset)
            indices = np.random.choice(len(remaining), min(additional, len(remaining)), replace=False)
            subset.extend([remaining[i] for i in indices])
    
    return subset[:n_samples]

def format_training_example(example: Dict, metadata_handler: EnhancedMetadataHandler) -> Dict:
    """Format training example with metadata."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    metadata = example.get("metadata", {})
    
    enriched_prompt = metadata_handler.enrich(instruction, metadata)
    
    text = (
        f"{enriched_prompt}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Output:\n{output_text}"
    )
    
    return {"text": text}

# ============================================================================
# CUSTOM DATA COLLATOR WITH PROPER LABEL MASKING (CRITICAL FIX)
# ============================================================================
class LabelMaskingDataCollator:
    """Data collator that masks labels for prompt and input, keeping only output."""
    
    def __init__(self, tokenizer, output_marker: str = "### Output:"):
        self.tokenizer = tokenizer
        self.output_marker = output_marker
    
    def __call__(self, features):
        # First, pad the inputs
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels copy
        labels = batch["input_ids"].clone()
        
        # Mask everything except the output part
        for i, input_ids in enumerate(batch["input_ids"]):
            # Convert tokens to string to find output marker
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Find output marker
            output_start = text.find(self.output_marker)
            if output_start == -1:
                # No output marker found, mask everything
                labels[i] = -100
                continue
            
            # Tokenize the text up to output marker to find position
            pre_output = text[:output_start + len(self.output_marker)]
            pre_output_tokens = self.tokenizer.encode(pre_output, add_special_tokens=False)
            
            # Account for special tokens
            special_tokens_count = len(self.tokenizer.encode("", add_special_tokens=True)) - 1
            
            # Output starts after the marker (+1 for newline)
            output_start_idx = len(pre_output_tokens) + special_tokens_count
            
            # Mask everything before output
            labels[i, :output_start_idx] = -100
            
            # Also mask padding tokens
            if "attention_mask" in batch:
                labels[i, batch["attention_mask"][i] == 0] = -100
        
        batch["labels"] = labels
        return batch

# ============================================================================
# MAIN TRAINING PIPELINE WITH ALL OPTIMIZATIONS
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="OPTIMIZED Option 4 Pipeline with ALL "
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--model_size", type=str, default="7b",
                       choices=["1.3b", "3b", "7b", "auto"],
                       help="Model size (auto detects from name)")
    parser.add_argument("--metadata", type=str, default="none",
                       choices=["none", "full", "structured", "algorithmic", "all_json"],
                       help="Metadata strategy")
    parser.add_argument("--output_dir", type=str, default="results_option4_optimized",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens (384 for 1.3B, 512 for 7B)")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience (increased from 2)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation after training")
    parser.add_argument("--run_cross_validation", action="store_true",
                       help="Run 5-fold cross-validation as supplementary")
    
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMIZED OPTION 4 PIPELINE ")
    print("="*80)
    print("Key Fixes Implemented:")
    print("  1. Label masking - loss only on Isabelle output")
    print("  2. Higher LoRA rank for 1.3B (64)")
    print("  3. No repetition penalty for evaluation")
    print("  4. Model-specific LoRA optimizations")
    print("  5. Early stopping with patience=3")
    print("  6. Complete debugging and analysis")
    print("="*80)
    
    from transformers import (
        Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM,
        DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # ========== LOAD DATA ==========
    print("\nLOADING DATA...")
    all_data, lea_data, rectangle_data, xtea_data, training_subset = load_all_training_data()
    
    if not all_data:
        print("ERROR: No training data found!")
        return
    
    # ========== DETECT MODEL TYPE ==========
    model_path = Path(args.model)
    model_name = model_path.name.lower()
    
    # Auto-detect model size if needed
    if args.model_size == "auto":
        if "1.3" in model_name or "1-3" in model_name:
            args.model_size = "1.3b"
        elif "3b" in model_name and "7" not in model_name and "13" not in model_name:
            args.model_size = "3b"
        else:
            args.model_size = "7b"
    
    print(f"Model: {model_path.name}")
    print(f"Detected size: {args.model_size}")
    
    # Detect model family for optimizations
    model_family = "unknown"
    if "qwen" in model_name:
        model_family = "qwen"
    elif "starcoder" in model_name:
        model_family = "starcoder"
    elif "deepseek" in model_name:
        model_family = "deepseek"
    elif "llama" in model_name:
        model_family = "llama"
    
    # ========== GET OPTIMIZED CONFIG ==========
    data_config = get_data_driven_config(args.model_size, model_family)
    
    # Override with command line
    if args.num_epochs:
        data_config["num_epochs"] = args.num_epochs
    if args.learning_rate:
        data_config["learning_rate"] = args.learning_rate
    
    print("\nOPTIMIZED CONFIGURATION:")
    print(f"  Metadata: {args.metadata}")
    print(f"  Epochs: {data_config['num_epochs']}")
    print(f"  Batch: {data_config['batch_size']} × {data_config['gradient_accumulation']}")
    print(f"  Learning rate: {data_config['learning_rate']:.1e}")
    print(f"  LoRA rank: {data_config.get('lora_rank', 'N/A')}")
    print(f"  4-bit: {data_config['use_4bit']}")
    
    # ========== CREATE OUTPUT DIRECTORY ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_path.stem}_{args.metadata}_{args.model_size}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOUTPUT: {output_dir}")
    
    # ========== FORMAT DATASETS ==========
    print("\nFORMATTING DATASETS...")
    metadata_handler = EnhancedMetadataHandler(args.metadata)
    
    def format_dataset(data: List[Dict]) -> Dataset:
        formatted = []
        for example in tqdm(data, desc="Formatting"):
            formatted.append(format_training_example(example, metadata_handler))
        return Dataset.from_list(formatted)
    
    train_dataset = format_dataset(all_data)
    lea_dataset = format_dataset(lea_data) if lea_data else None
    
    print(f"  Training: {len(train_dataset)} examples")
    if lea_dataset:
        print(f"  LEA validation: {len(lea_dataset)} examples")
    
    # ========== LOAD TOKENIZER ==========
    print("\nLOADING TOKENIZER...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Debug: Check tokenization
    if args.debug and len(train_dataset) > 0:
        print("\n=== TOKENIZATION DEBUG ===")
        sample_text = train_dataset[0]["text"]
        tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        print(f"Sample text length: {len(sample_text)} chars")
        print(f"Tokenized length: {len(tokens)} tokens")
        print(f"First 10 tokens: {tokens[:10]}")
        
        # Check for output marker
        if "### Output:" in sample_text:
            print("Output marker found in sample")
    
    # ========== TOKENIZE WITH LABEL MASKING ==========
    print("\nTOKENIZING WITH LABEL MASKING...")
    
    def tokenize_function(examples):
        """Tokenize without padding."""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config["max_length"],
            padding=False,  # Will pad in collator
            return_tensors=None,
        )
        return result
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    if lea_dataset:
        tokenized_lea = lea_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Debug label alignment
    if args.debug and len(tokenized_train) > 0:
        TrainingDebugger.check_label_alignment(tokenizer, tokenized_train[0])
    
    # ========== LOAD MODEL ==========
    print("\nLOADING MODEL...")
    
    is_deepseek_v2 = "lite" in args.model_size.lower() or "v2" in args.model_size.lower()
    
    if data_config["use_4bit"]:
        print("  Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,
        )
    else:
        print("  Using FP16 precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,
        )
    
    if data_config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)
    
    # ========== APPLY LoRA WITH MODEL-SPECIFIC OPTIMIZATIONS ==========
    print("\nAPPLYING LoRA...")
    
    lora_config = LoraConfig(
        r=data_config.get("lora_rank", 16),
        lora_alpha=data_config.get("lora_alpha", 32),
        target_modules=data_config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=data_config.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100
    
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable %: {trainable_pct:.2f}%")
    
    # Check if percentage is reasonable
    if trainable_pct > 2.0 and "7b" in args.model_size:
        print(f"  ⚠ Warning: High trainable % for 7B model")
        print(f"  Consider reducing LoRA rank further")
    
    # ========== SETUP TRAINING ==========
    print("\nCONFIGURING TRAINING...")
    
    # Calculate steps
    total_steps = len(tokenized_train) // (data_config["batch_size"] * data_config["gradient_accumulation"]) * data_config["num_epochs"]
    eval_steps = max(25, total_steps // 20)  # Evaluate ~20 times
    save_steps = eval_steps * 2
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=data_config["num_epochs"],
        per_device_train_batch_size=data_config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=data_config["gradient_accumulation"],
        learning_rate=data_config["learning_rate"],
        fp16=not data_config["use_4bit"] and not is_deepseek_v2,
        bf16=is_deepseek_v2,
        logging_steps=10,
        eval_strategy="steps" if lea_dataset else "no",
        eval_steps=eval_steps if lea_dataset else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True if lea_dataset else False,
        metric_for_best_model="eval_loss" if lea_dataset else None,
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=not is_deepseek_v2 and ("7b" in args.model_size or "6.7b" in args.model_size),
        optim="adamw_torch",
        warmup_ratio=data_config["warmup_ratio"],
        weight_decay=data_config["weight_decay"],
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    
    # ========== CREATE TRAINER WITH LABEL MASKING ==========
    print("\nCREATING TRAINER WITH LABEL MASKING...")
    
    # Use custom collator for label masking
    data_collator = LabelMaskingDataCollator(tokenizer)
    
    # Early stopping callback
    callbacks = []
    if lea_dataset and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.001
        ))
        print(f"  Early stopping patience: {args.early_stopping_patience}")
    
    if lea_dataset:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_lea,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        print(f"  Training with LEA validation")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        print(f"  Training without validation")
    
    # ========== TRAIN ==========
    print("\n" + "="*80)
    print("STARTING OPTIMIZED TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time/60:.1f} minutes")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        if lea_dataset:
            eval_results = trainer.evaluate()
            print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
    
    except Exception as e:
        print(f"ERROR during training: {e}")
        print("Saving partial model...")
        trainer.save_model(str(output_dir / "partial_model"))
        raise
    
    # ========== SAVE FINAL MODEL ==========
    print("\nSAVING FINAL MODEL...")
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save training configuration
    config_summary = {
        "model": str(model_path),
        "model_size": args.model_size,
        "metadata_strategy": args.metadata,
        "training_config": data_config,
        "training_args": {
            "num_epochs": data_config["num_epochs"],
            "batch_size": data_config["batch_size"],
            "gradient_accumulation": data_config["gradient_accumulation"],
            "learning_rate": data_config["learning_rate"],
            "lora_rank": data_config.get("lora_rank", "N/A"),
            "lora_alpha": data_config.get("lora_alpha", "N/A"),
            "early_stopping_patience": args.early_stopping_patience,
        },
        "model_stats": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_pct,
        },
        "data_stats": {
            "training_examples": len(all_data),
            "lea_validation": len(lea_data) if lea_data else 0,
            "rectangle_test": len(rectangle_data) if rectangle_data else 0,
            "xtea_test": len(xtea_data) if xtea_data else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(final_model_dir / "training_config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"  Model saved to: {final_model_dir}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    # ========== EVALUATION ==========
    if args.skip_evaluation:
        print("\nSkipping evaluation as requested")
        return
    
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)
    
    # Import evaluator here to avoid circular imports
    
    
    def evaluate_with_full_results(dataset: List[Dict], dataset_name: str, model, tokenizer, metadata_handler) -> Dict:
        """Full evaluation with generated text and scores."""
        print(f"\nEvaluating {dataset_name} ({len(dataset)} examples)...")
        
        model.eval()
        optimize_model_for_evaluation(model, data_config["use_4bit"])
        
        results = {
            "dataset": dataset_name,
            "total_examples": len(dataset),
            "detailed_results": []
        }
        
        # Generate for each example
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            # Prepare prompt
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            metadata = example.get("metadata", {})
            
            enriched_prompt = metadata_handler.enrich(instruction, metadata)
            prompt = f"{enriched_prompt}\n\n### Input:\n{input_text}\n\n### Output:"
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                )
            
            # Decode
            generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            generated = extract_isabelle_code(generated)
            reference = clean_code(example.get("output", ""))
            
            # Store everything
            results["detailed_results"].append({
                "index": i,
                "prompt": prompt,
                "generated_full": generated,
                "generated_truncated": generated[:600] + "..." if len(generated) > 600 else generated,
                "reference_full": reference,
                "reference_truncated": reference[:600] + "..." if len(reference) > 600 else reference,
                "metadata": metadata,
                "cipher": metadata.get("cipher", "unknown"),
                "family": metadata.get("family", "unknown"),
                "difficulty": metadata.get("difficulty", "medium"),
                # Placeholder for scores (will be computed separately)
                "sv": 0.0,
                "sm": 0.0,
                "vc": 0.0,
                "overall": 0.0,
            })
        
        return results

        
    ######## Run evaluations with full results
    all_results = {}
    
    if training_subset:
        print('training_subset')
        results = evaluate_with_full_results(training_subset, "training_subset", model, tokenizer, metadata_handler)
        all_results["training_subset"] = results
        results = compute_metrics_for_results(results, metrics)
        save_json(results, output_dir / "training_subset_full_results.json")
        
    if rectangle_data:
        print('rectangle_data')
        results = evaluate_with_full_results(rectangle_data, "rectangle", model, tokenizer, metadata_handler)
        all_results["rectangle"] = results
        results = compute_metrics_for_results(results, metrics)
        save_json(results, output_dir / "rectangle_full_results.json")
    
    if xtea_data:
        print('xtea_data')
        results = evaluate_with_full_results(xtea_data, "xtea", model, tokenizer, metadata_handler)
        all_results["xtea"] = results
        results = compute_metrics_for_results(results, metrics)
        save_json(results, output_dir / "xtea_full_results.json")
    
    if lea_data:
        print('lea_data')
        results = evaluate_with_full_results(lea_data, "lea", model, tokenizer, metadata_handler)
        all_results["lea"] = results
        results = compute_metrics_for_results(results, metrics)
        save_json(results, output_dir / "lea_full_results.json")

    # ========== RUN CROSS-VALIDATION (SUPPLEMENTARY) ==========
    if args.run_cross_validation and args.model_size in ['1.3b', '3b']:
        print("\n" + "="*80)
        print(f"RUNNING 3-FOLD CROSS-VALIDATION (SUPPLEMENTARY) for {args.model_size} MODEL SIZE")
        print("="*80)
        
        # Use 3-fold instead of 5-fold (faster, still meaningful)
        folds = create_cross_validation_folds(all_data, n_folds=3)
        cv_results = []
        
        for fold_idx, (train_fold, val_fold) in enumerate(folds, 1):
            print(f"\nFold {fold_idx}/{len(folds)}: "
                  f"Train={len(train_fold)}, Val={len(val_fold)}")
            
            # Count families for stratification analysis
            train_families = {}
            val_families = {}
            
            for example in train_fold:
                family = example.get("metadata", {}).get("family", "unknown")
                train_families[family] = train_families.get(family, 0) + 1
            
            for example in val_fold:
                family = example.get("metadata", {}).get("family", "unknown")
                val_families[family] = val_families.get(family, 0) + 1
            
            cv_results.append({
                "fold": fold_idx,
                "train_size": len(train_fold),
                "val_size": len(val_fold),
                "train_families": train_families,
                "val_families": val_families,
                # Add difficulty distribution too
                "train_difficulties": count_difficulties(train_fold),
                "val_difficulties": count_difficulties(val_fold),
            })
        
        cv_file = output_dir / "cross_validation_summary.json"
        with open(cv_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"  CV summary saved to: {cv_file}")
        
        # Print simple statistics
        print(f"\nCV Statistics (3-fold):")
        print(f"{'Fold':<6} {'Train':<8} {'Val':<8} {'Train/Val Ratio':<15}")
        print("-" * 40)
        for result in cv_results:
            ratio = result["train_size"] / result["val_size"] if result["val_size"] > 0 else 0
            print(f"{result['fold']:<6} {result['train_size']:<8} {result['val_size']:<8} {ratio:<15.2f}")
            
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("OPTIMIZED PIPELINE COMPLETE!")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Size: {args.model_size}")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {output_dir}")
    print(f"Final model: {final_model_dir}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    print("\nNEXT STEPS:")
    print(f"1. Run full evaluation: python scripts/evaluate_icml_pipeline_curriculum.py \\")
    print(f"     --model_path {final_model_dir} \\")
    print(f"     --metadata_strategy {args.metadata}")
    print(f"2. Check logs in: {output_dir}")
    print(f"3. Compare with baseline models")
    print("="*80)

def count_families(data: List[Dict]) -> Dict[str, int]:
    """Count cipher families in dataset."""
    families = {}
    for example in data:
        family = example.get("metadata", {}).get("family", "unknown")
        families[family] = families.get(family, 0) + 1
    return families

def count_difficulties(data: List[Dict]) -> Dict[str, int]:
    """Count difficulty levels in dataset."""
    difficulties = {}
    for example in data:
        difficulty = example.get("metadata", {}).get("difficulty", "medium")
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    return difficulties
        
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_metrics_for_results(results: Dict, metrics) -> Dict:
    """Compute SV, SM, VC scores for generated results."""
    if metrics is None:
        print("Warning: No metrics available, skipping computation")
        return results
    
    for item in results["detailed_results"]:
        try:
            metrics_result = metrics.evaluate(item["generated_full"], item["reference_full"])
            item["sv"] = metrics_result.get("syntax_validity", 0.0)
            item["sm"] = metrics_result.get("semantic_match", 0.0)
            item["vc"] = metrics_result.get("value_consistency", 0.0)
            item["overall"] = (item["sv"] + item["sm"] + item["vc"]) / 3.0

            print(f"SV: item[sv] : ", item["sv"])
            print(f"SM: item[sm] : ", item["sm"])
            print(f"VC: item[vc] : ", item["vc"])
            print(f"overall: item[overall] : ", item["overall"])
            
        except Exception as e:
            print(f"Error computing metrics for example {item['index']}: {e}")

    
    # Compute averages
    if results["detailed_results"]:
        results["avg_sv"] = np.mean([item["sv"] for item in results["detailed_results"]])
        results["avg_sm"] = np.mean([item["sm"] for item in results["detailed_results"]])
        results["avg_vc"] = np.mean([item["vc"] for item in results["detailed_results"]])
        results["avg_overall"] = np.mean([item["overall"] for item in results["detailed_results"]])
    
    return results

def save_json(data: Dict, filepath: Path):
    """Save data to JSON file."""
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(convert_numpy(data), f, indent=2)

# ============================================================================
if __name__ == "__main__":
    main()
