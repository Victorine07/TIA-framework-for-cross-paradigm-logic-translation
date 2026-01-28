#!/usr/bin/env python3
"""
STUDY II: OOD Training for DeepSeek-Coder-V2-Lite (MoE Model)
Enhanced OOD Generalization (The TIA Protocol)
Training on entire dataset, validation on LEA (unseen), test on RECTANGLE/XTEA
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
from transformers import BitsAndBytesConfig
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# PATH SETUP
# ============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
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
# DEEPSEEK V2 SPECIFIC CONFIGURATION FOR OOD TRAINING
# ============================================================================
def get_deepseek_v2_ood_config() -> dict:
    """
    Specialized configuration for DeepSeek-Coder-V2-Lite (MoE model) for OOD training
    """
    config = {
        "max_length": 1024,
        "num_epochs": 4,  # Same as Study I for comparison
        "batch_size": 1,  # Conservative due to MoE
        "gradient_accumulation": 16,  # Increased for stable training
        "learning_rate": 2e-5,  # Slightly lower for MoE
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "use_4bit": True,  # Using 4-bit quantization as in your working script
        "use_bf16": False,  # Using float16 with 4-bit
        "gradient_checkpointing": True,  # Critical for MoE models
        "lora_rank": 16,    # Conservative rank for MoE
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", 
            "kv_b_proj", "o_proj", "gate_proj"
        ],  # DeepSeek-V2 specific modules from your working config
        "model_type": "deepseek_v2",
    }
    
    print(f"\nDEEPSEEK-V2-LITE OOD CONFIG:")
    print(f"  Using 4-bit quantization with float16 compute")
    print(f"  Batch size: {config['batch_size']} × {config['gradient_accumulation']}")
    print(f"  Gradient checkpointing: Enabled")
    print(f"  LoRA target: {config['target_modules']}")
    print(f"  OOD Protocol: Train on all data, validate on LEA (unseen)")
    
    return config

# ============================================================================
# METADATA HANDLER (IDENTICAL TO OTHER MODELS)
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
        elif self.strategy == "all_json":
            return self._enrich_all_json(instruction, metadata, constraint)
        else:
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
        params = metadata.get("algorithm_params", {})
        variant = metadata.get("variant", {})
        cipher_name = metadata.get("cipher", "Unknown")
        family = metadata.get("family", "Unknown")
        
        if not params and not variant:
            return f"### Task: {instruction}{constraint}"
        
        priority_params = [
            ("Cipher", f"{cipher_name} ({family})"),
            ("Block Size", variant.get("block_size", "N/A")),
            ("Word Size", params.get("word_size", variant.get("word_size", "N/A")))
        ]
        
        # Family-specific logic
        if family in ["ARX", "Feistel"]:
            for key in ["alpha_rotation", "beta_rotation", "rotation_constants"]:
                if key in params:
                    priority_params.append((key.replace("_", " ").title(), params[key]))
            if "arx_order" in params:
                priority_params.append(("Operation Sequence", params["arx_order"]))
            if "f_function" in params:
                priority_params.append(("F-Function Logic", params["f_function"]))
        
        elif family == "SPN":
            if "sbox_size" in params:
                priority_params.append(("S-Box Size", f"{params['sbox_size']}-bit"))
            if "sbox_count" in params:
                priority_params.append(("S-Box Count", params["sbox_count"]))
            if "permutation_type" in params:
                priority_params.append(("Permutation", params["permutation_type"]))
            if "nibbles_per_block" in variant:
                priority_params.append(("Nibble Count", variant["nibbles_per_block"]))
        
        if "rounds" in variant:
            priority_params.append(("Total Rounds", variant["rounds"]))
        
        comp_type = metadata.get("component_type", "")
        if comp_type:
            priority_params.append(("Component Tier", comp_type))
        
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
        
        if "word_size" in params:
            core_info.append(f"{params['word_size']}b")
        
        if "arx_order" in params:
            order = params["arx_order"]
            if "→" in order:
                core_info.append(order.split("→")[0].strip())
            else:
                core_info.append("ARX")
        elif "f_function" in params:
            f_func = str(params['f_function'])
            if "<<" in f_func:
                core_info.append("rotate")
            if "&" in f_func:
                core_info.append("and")
            if "^" in f_func:
                core_info.append("xor")
        
        if core_info:
            spec_line = " + ".join(core_info[:3])
            return f"### Algorithm: {spec_line}\n\n### Task: {instruction}{constraint}"
        
        return f"### Task: {instruction}{constraint}"

    def _enrich_all_json(self, instruction: str, metadata: Dict, constraint: str) -> str:
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_metadata = flatten_dict(metadata)
        spec_lines = []
        for key, value in flat_metadata.items():
            clean_key = key.replace('_', ' ').replace('.', ' → ').title()
            spec_lines.append(f"- {clean_key}: {value}")
        
        spec_block = "\n".join(spec_lines)
        return f"### Complete Metadata Specification:\n{spec_block}\n\n### Task: {instruction}{constraint}"

# ============================================================================
# DATA PROCESSING FOR OOD TRAINING
# ============================================================================
def load_ood_training_data():
    """
    Load data for OOD Training (Study II):
    - All training data (train + val + test) for training
    - LEA for validation (unseen cipher)
    - RECTANGLE and XTEA for testing (unseen ciphers)
    """
    processed_dir = Path("data/processed")
    unseen_dir = Path("data/unseen")
    
    # Load all data for training
    train_data = load_jsonl(processed_dir / "train.jsonl")
    val_data = load_jsonl(processed_dir / "val.jsonl")
    test_data = load_jsonl(processed_dir / "test.jsonl")
    
    # Combine all data for OOD training
    all_training_data = train_data + val_data + test_data
    
    # Load unseen datasets
    lea_data = load_jsonl(unseen_dir / "lea_dataset.jsonl")
    rectangle_data = load_jsonl(unseen_dir / "rectangle_dataset.jsonl")
    xtea_data = load_jsonl(unseen_dir / "xtea_dataset.jsonl")
    
    print(f"\nOOD TRAINING DATA SUMMARY:")
    print(f"  All training data: {len(all_training_data)} examples")
    print(f"    (Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)})")
    print(f"\n  Unseen ciphers for OOD evaluation:")
    print(f"    LEA (validation): {len(lea_data)} examples")
    print(f"    RECTANGLE (test): {len(rectangle_data)} examples")
    print(f"    XTEA (test): {len(xtea_data)} examples")
    
    # Count families in training data
    print(f"\n  Training data families:")
    train_families = {}
    for example in all_training_data:
        family = example.get("metadata", {}).get("family", "unknown")
        train_families[family] = train_families.get(family, 0) + 1
    for family, count in train_families.items():
        print(f"    {family}: {count} examples")
    
    return all_training_data, lea_data, rectangle_data, xtea_data

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

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
# CUSTOM DATA COLLATOR WITH LABEL MASKING
# ============================================================================
class LabelMaskingDataCollator:
    """Data collator that masks labels for prompt and input, keeping only output."""
    
    def __init__(self, tokenizer, output_marker: str = "### Output:"):
        self.tokenizer = tokenizer
        self.output_marker = output_marker
    
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        labels = batch["input_ids"].clone()
        
        for i, input_ids in enumerate(batch["input_ids"]):
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            output_start = text.find(self.output_marker)
            
            if output_start == -1:
                labels[i] = -100
                continue
            
            pre_output = text[:output_start + len(self.output_marker)]
            pre_output_tokens = self.tokenizer.encode(pre_output, add_special_tokens=False)
            special_tokens_count = len(self.tokenizer.encode("", add_special_tokens=True)) - 1
            output_start_idx = len(pre_output_tokens) + special_tokens_count
            
            labels[i, :output_start_idx] = -100
            
            if "attention_mask" in batch:
                labels[i, batch["attention_mask"][i] == 0] = -100
        
        batch["labels"] = labels
        return batch

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_isabelle_code(text: str) -> str:
    """Extract Isabelle/HOL code from generated text."""
    text = re.sub(r'```\w*\n', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    patterns = [
        r'(definition\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
        r'(fun\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
        r'(lemma\s+\w+\s*(::|where).*?)(?=\n\s*(?:definition|fun|lemma|end|\Z))',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0][0].strip()
            lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped:
                    if stripped.startswith(('definition', 'fun', 'lemma')):
                        if not (stripped.endswith('where') or '::' in stripped):
                            continue
                    lines.append(line)
            if lines:
                return '\n'.join(lines)
    
    lines = []
    in_isabelle_block = False
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.startswith(('definition', 'fun', 'lemma')):
            in_isabelle_block = True
            lines.append(line)
        elif in_isabelle_block:
            if re.match(r'^\s*(?:definition|fun|lemma|where|::|=|"|\||end)', stripped):
                lines.append(line)
            else:
                break
        elif '::' in stripped or 'where' in stripped:
            lines.append(line)
    
    if lines:
        return '\n'.join(lines)
    
    return clean_code(text)

def clean_code(code_str: str) -> str:
    """Clean Isabelle/HOL code."""
    import re
    code_str = re.sub(r'\(\*.*?\*\)', '', code_str, flags=re.DOTALL)
    code_str = re.sub(r'#.*', '', code_str)
    lines = [line.rstrip() for line in code_str.splitlines() if line.strip()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    return '\n'.join(lines)

# ============================================================================
# MAIN TRAINING PIPELINE FOR DEEPSEEK-V2-LITE (OOD TRAINING)
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="STUDY II: DeepSeek-Coder-V2-Lite OOD Training (TIA Protocol)"
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to DeepSeek-Coder-V2-Lite directory")
    parser.add_argument("--metadata", type=str, default="structured",
                       choices=["none", "full", "structured", "algorithmic", "all_json"],
                       help="Metadata strategy")
    parser.add_argument("--output_dir", type=str, default="results_deepseek_v2_ood",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=4,
                       help="Number of epochs (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--gradient_accumulation", type=int, default=16,
                       help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank (default: 16)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens for generation")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience (default: 3)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation after training")
    
    args = parser.parse_args()
    
    print("="*80)
    print("STUDY II: DEEPSEEK-CODER-V2-LITE OOD TRAINING (TIA PROTOCOL)")
    print("="*80)
    print("OOD Protocol Characteristics:")
    print("  1. Train on ENTIRE dataset (train + val + test)")
    print("  2. Validate on LEA (completely unseen cipher)")
    print("  3. Test on RECTANGLE and XTEA (unseen ciphers)")
    print("  4. Maximizes gradient signal with generalization guardrail")
    print("="*80)
    print("Model: DeepSeek-Coder-V2-Lite (MoE architecture)")
    print("  - Using 4-bit quantization with float16 compute")
    print("  - Gradient checkpointing enabled")
    print("  - Conservative LoRA configuration")
    print("="*80)
    
    # ========== LOAD DATA FOR OOD TRAINING ==========
    print("\nLOADING OOD TRAINING DATA...")
    all_training_data, lea_data, rectangle_data, xtea_data = load_ood_training_data()
    
    if not all_training_data:
        print("ERROR: No training data found!")
        return
    
    # ========== CONFIGURATION ==========
    config = get_deepseek_v2_ood_config()
    
    # Override with command line args
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.gradient_accumulation:
        config["gradient_accumulation"] = args.gradient_accumulation
    if args.lora_rank:
        config["lora_rank"] = args.lora_rank
    
    print(f"\nFINAL OOD CONFIGURATION:")
    print(f"  Metadata: {args.metadata}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch: {config['batch_size']} × {config['gradient_accumulation']}")
    print(f"  Learning rate: {config['learning_rate']:.1e}")
    print(f"  LoRA rank: {config['lora_rank']}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    
    # ========== CREATE OUTPUT DIRECTORY ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"deepseek_v2_ood_{args.metadata}_{timestamp}"
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
    
    # Format all datasets
    train_dataset = format_dataset(all_training_data)
    lea_dataset = format_dataset(lea_data) if lea_data else None
    
    print(f"  Training (all data): {len(train_dataset)} examples")
    if lea_dataset:
        print(f"  Validation (LEA): {len(lea_dataset)} examples")
    
    # ========== LOAD TOKENIZER ==========
    print("\nLOADING DEEPSEEK-V2 TOKENIZER...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # ========== TOKENIZE ==========
    print("\nTOKENIZING...")
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_length"],
            padding=False,
            return_tensors=None,
        )
        return result
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    if lea_dataset:
        tokenized_lea = lea_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # ========== LOAD DEEPSEEK-V2 MODEL ==========
    print("\nLOADING DEEPSEEK-V2-LITE MODEL (MoE)...")
    
    # Define 4-bit config (as in your working script)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        attn_implementation="eager",
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
        model.enable_input_require_grads()
    
    # ========== APPLY LoRA ==========
    print("\nAPPLYING LoRA TO DEEPSEEK-V2...")
    
    # Use DeepSeek-V2 specific modules from your working config
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Attach the adapters
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing explicitly
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100
    
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable %: {trainable_pct:.2f}%")
    
    # ========== SETUP TRAINING ==========
    print("\nCONFIGURING TRAINING...")
    
    total_steps = len(tokenized_train) // (config["batch_size"] * config["gradient_accumulation"]) * config["num_epochs"]
    eval_steps = max(25, total_steps // 20)
    save_steps = eval_steps * 2
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        bf16=config["use_bf16"],
        fp16=False,
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
        gradient_checkpointing=config["gradient_checkpointing"],
        optim="adamw_torch",
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    
    # ========== CREATE TRAINER ==========
    print("\nCREATING TRAINER...")
    data_collator = LabelMaskingDataCollator(tokenizer)
    
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
        print(f"  Training with LEA validation (unseen cipher)")
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
    print("STARTING DEEPSEEK-V2-LITE OOD TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time/60:.1f} minutes")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        if lea_dataset:
            eval_results = trainer.evaluate()
            print(f"Validation loss (LEA): {eval_results['eval_loss']:.4f}")
    
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
    
    # Save config
    config_summary = {
        "study": "Study II: OOD Training (TIA Protocol)",
        "model": str(args.model),
        "metadata_strategy": args.metadata,
        "training_config": config,
        "early_stopping_patience": args.early_stopping_patience,
        "model_stats": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_pct,
        },
        "data_stats": {
            "training_examples": len(all_training_data),
            "validation_examples": len(lea_data) if lea_data else 0,
            "test_examples_rectangle": len(rectangle_data) if rectangle_data else 0,
            "test_examples_xtea": len(xtea_data) if xtea_data else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(final_model_dir / "training_config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"  Model saved to: {final_model_dir}")
    
    # ========== EVALUATION ==========
    if args.skip_evaluation:
        print("\nSkipping evaluation as requested")
        return
    
    print("\n" + "="*80)
    print("OOD EVALUATION PHASE")
    print("="*80)
    print("Testing on unseen ciphers:")
    print("  1. RECTANGLE (completely unseen)")
    print("  2. XTEA (completely unseen)")
    print("="*80)
    
    def evaluate_dataset(dataset: List[Dict], dataset_name: str):
        """Evaluate on a specific dataset."""
        print(f"\nEvaluating {dataset_name} ({len(dataset)} examples)...")
        
        model.eval()
        results = []
        
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            metadata = example.get("metadata", {})
            
            enriched_prompt = metadata_handler.enrich(instruction, metadata)
            prompt = f"{enriched_prompt}\n\n### Input:\n{input_text}\n\n### Output:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )
            
            generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            generated = extract_isabelle_code(generated)
            reference = clean_code(example.get("output", ""))
            
            # Compute metrics if available
            sv = sm = vc = overall = 0.0
            if metrics:
                try:
                    metrics_result = metrics.evaluate(generated, reference)
                    sv = metrics_result.get("syntax_validity", 0.0)
                    sm = metrics_result.get("semantic_match", 0.0)
                    vc = metrics_result.get("value_consistency", 0.0)
                    overall = (sv + sm + vc) / 3.0
                except Exception as e:
                    print(f"Error computing metrics: {e}")
            
            results.append({
                "generated": generated,
                "reference": reference,
                "metrics": {"sv": sv, "sm": sm, "vc": vc, "overall": overall}
            })
        
        # Compute averages
        if results:
            avg_sv = np.mean([r["metrics"]["sv"] for r in results])
            avg_sm = np.mean([r["metrics"]["sm"] for r in results])
            avg_vc = np.mean([r["metrics"]["vc"] for r in results])
            avg_overall = np.mean([r["metrics"]["overall"] for r in results])
            
            print(f"  Average - SV: {avg_sv:.3f}, SM: {avg_sm:.3f}, VC: {avg_vc:.3f}, Overall: {avg_overall:.3f}")
        
        # Save results
        results_file = output_dir / f"{dataset_name}_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "count": len(results),
                "average_metrics": {
                    "sv": float(avg_sv) if results else 0.0,
                    "sm": float(avg_sm) if results else 0.0,
                    "vc": float(avg_vc) if results else 0.0,
                    "overall": float(avg_overall) if results else 0.0,
                },
                "results": results
            }, f, indent=2)
        
        return avg_sv, avg_sm, avg_vc, avg_overall
    
    # Run OOD evaluations
    eval_results = {}
    
    if rectangle_data:
        eval_results["rectangle"] = evaluate_dataset(rectangle_data, "rectangle")
    
    if xtea_data:
        eval_results["xtea"] = evaluate_dataset(xtea_data, "xtea")
    
    # Also evaluate on LEA for completeness
    if lea_data:
        eval_results["lea"] = evaluate_dataset(lea_data, "lea")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("DEEPSEEK-V2-LITE OOD TRAINING COMPLETE!")
    print("="*80)
    print(f"Study: II - OOD Generalization (TIA Protocol)")
    print(f"Model: DeepSeek-Coder-V2-Lite (MoE)")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {output_dir}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    print("\nOOD EVALUATION SUMMARY:")
    if eval_results.get("lea"):
        sv, sm, vc, overall = eval_results["lea"]
        print(f"  LEA (validation)    - SV: {sv:.3f}, SM: {sm:.3f}, VC: {vc:.3f}, Overall: {overall:.3f}")
    if eval_results.get("rectangle"):
        sv, sm, vc, overall = eval_results["rectangle"]
        print(f"  RECTANGLE (test)    - SV: {sv:.3f}, SM: {sm:.3f}, VC: {vc:.3f}, Overall: {overall:.3f}")
    if eval_results.get("xtea"):
        sv, sm, vc, overall = eval_results["xtea"]
        print(f"  XTEA (test)         - SV: {sv:.3f}, SM: {sm:.3f}, VC: {vc:.3f}, Overall: {overall:.3f}")
    
    print("\nCOMPARISON WITH STUDY I:")
    print("  This OOD training should provide better generalization to unseen ciphers")
    print("  compared to the 80-10-10 split in Study I.")
    
    print("\nNEXT STEPS:")
    print(f"1. Compare OOD results with Study I baseline")
    print(f"2. Analyze generalization patterns across cipher families")
    print(f"3. Check detailed results in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
