#!/usr/bin/env python3
"""
Zero-shot evaluation for DeepSeek-Coder-V2-Lite on OOD datasets
Evaluates untrained DeepSeek-V2-Lite on LEA, RECTANGLE, and XTEA
"""

import json
import torch
import argparse
from pathlib import Path
import sys
import re
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import os

# Add project paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import metrics
try:
    from evaluation.metrics_fixed import FixedCryptographicMetrics
    print("✓ Using FixedCryptographicMetrics")
    METRICS_CLASS = FixedCryptographicMetrics
except ImportError:
    print("✗ FixedCryptographicMetrics not found, using basic evaluation")
    METRICS_CLASS = None

from json import JSONEncoder
import numpy as np
import torch

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

# ============================================================================
# DEEPSEEK-V2-LITE SPECIFIC CONFIGURATION
# ============================================================================
def get_deepseek_v2_config() -> Dict:
    """Configuration optimized for DeepSeek-Coder-V2-Lite."""
    return {
        "max_length": 1024,
        "max_new_tokens": 512,
        "use_4bit": True,
        "compute_dtype": torch.float16,
        "attn_implementation": "eager",
        "lora_rank": 16,  # Not used for zero-shot, but for reference
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }

def clean_code(code_str: str) -> str:
    """Clean Isabelle/HOL code."""
    # Remove comments
    code_str = re.sub(r'\(\*.*?\*\)', '', code_str, flags=re.DOTALL)
    code_str = re.sub(r'#.*', '', code_str)
    # Clean whitespace
    lines = [line.rstrip() for line in code_str.splitlines() if line.strip()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    return '\n'.join(lines)

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

# ============================================================================
# DEEPSEEK-V2-LITE ZERO-SHOT EVALUATOR
# ============================================================================
class DeepSeekV2ZeroShotEvaluator:
    """Evaluates DeepSeek-Coder-V2-Lite in zero-shot setting."""
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Args:
            model_path: Path to DeepSeek-Coder-V2-Lite directory
            config: Configuration dictionary (optional)
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        
        # Use provided config or default
        self.config = config or get_deepseek_v2_config()
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING DEEPSEEK-V2-LITE ZERO-SHOT EVALUATOR")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Path: {self.model_path}")
        print(f"Config:")
        print(f"  Max length: {self.config['max_length']}")
        print(f"  Max new tokens: {self.config['max_new_tokens']}")
        print(f"  4-bit quantization: {self.config['use_4bit']}")
        print(f"{'='*80}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model with 4-bit quantization
        print("Loading DeepSeek-V2-Lite model (4-bit)...")
        
        if self.config['use_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config['compute_dtype'],
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                use_cache=True,
                attn_implementation=self.config['attn_implementation'],
            )
        else:
            # Fallback to bfloat16 if 4-bit fails
            print("Warning: Using bfloat16 instead of 4-bit")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=True,
                attn_implementation=self.config['attn_implementation'],
            )
        
        # Initialize metrics
        if METRICS_CLASS:
            self.metrics = METRICS_CLASS()
        else:
            self.metrics = None
        
        print(f"✓ DeepSeek-V2-Lite loaded successfully")
        print(f"  Device: {self.model.device}")
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load JSONL dataset."""
        path = Path(filepath)
        if not path.exists():
            print(f"Error: Dataset file not found: {filepath}")
            return []
        
        with open(path, "r") as f:
            return [json.loads(line) for line in f]
    
    def create_zero_shot_prompt(self, example: Dict) -> str:
        """Create zero-shot prompt from example (no metadata)."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        
        # Simple zero-shot prompt format (no metadata)
        prompt = f"### Instruction: {instruction}\n### Input: {input_text}\n### Output:"
        return prompt
    
    def generate_response(self, example: Dict) -> str:
        """Generate response using zero-shot prompting."""
        prompt = self.create_zero_shot_prompt(example)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"],
            padding=True,
        )
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with deterministic settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False,  # Deterministic
                temperature=0.0,
                top_p=None,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,  # No repetition penalty
                use_cache=True,
            )
        
        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        
        if len(generated_ids) == 0:
            return ""
        
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated = extract_isabelle_code(generated)
        
        return generated
    
    def compute_metrics(self, generated: str, reference: str) -> Dict:
        """Compute SV, SM, VC metrics."""
        if not self.metrics:
            return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}
        
        try:
            results = self.metrics.evaluate(generated, reference)
            
            sv = results.get("syntax_validity", 0.0)
            sm = results.get("semantic_match", 0.0)
            vc = results.get("value_consistency", 0.0)
            overall = (sv + sm + vc) / 3.0
            
            return {
                "sv": float(sv),
                "sm": float(sm),
                "vc": float(vc),
                "overall": float(overall)
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}
    
    def evaluate_dataset(self, dataset: List[Dict], dataset_name: str, 
                         max_examples: Optional[int] = None) -> Dict:
        """Evaluate on a specific dataset."""
        if max_examples:
            dataset = dataset[:max_examples]
        
        print(f"\nEvaluating {dataset_name} ({len(dataset)} examples)...")
        
        results = {
            "model": self.model_name,
            "model_path": str(self.model_path),
            "dataset": dataset_name,
            "status": "zero_shot",
            "total_examples": len(dataset),
            "timestamp": datetime.now().isoformat(),
            "examples": [],
            "sv_scores": [],
            "sm_scores": [],
            "vc_scores": [],
            "overall_scores": [],
            "config": self.config,
        }
        
        # Process each example
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            # Generate response
            generated = self.generate_response(example)
            generated_clean = clean_code(generated)
            reference = clean_code(example.get("output", ""))
            
            # Compute metrics
            metrics = self.compute_metrics(generated_clean, reference)
            
            # Store scores
            results["sv_scores"].append(metrics["sv"])
            results["sm_scores"].append(metrics["sm"])
            results["vc_scores"].append(metrics["vc"])
            results["overall_scores"].append(metrics["overall"])
            
            # Store example details (truncated for file size)
            results["examples"].append({
                "index": i,
                "generated_truncated": generated_clean[:500] if len(generated_clean) > 500 else generated_clean,
                "reference_truncated": reference[:500] if len(reference) > 500 else reference,
                "sv": metrics["sv"],
                "sm": metrics["sm"],
                "vc": metrics["vc"],
                "overall": metrics["overall"]
            })
            
            # Optional: Print first few examples for debugging
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"  SV: {metrics['sv']:.3f}, SM: {metrics['sm']:.3f}, VC: {metrics['vc']:.3f}")
        
        # Calculate statistics
        if results["sv_scores"]:
            results["avg_sv"] = np.mean(results["sv_scores"])
            results["avg_sm"] = np.mean(results["sm_scores"])
            results["avg_vc"] = np.mean(results["vc_scores"])
            results["avg_overall"] = np.mean(results["overall_scores"])
            
            results["std_sv"] = np.std(results["sv_scores"])
            results["std_sm"] = np.std(results["sm_scores"])
            results["std_vc"] = np.std(results["vc_scores"])
            results["std_overall"] = np.std(results["overall_scores"])
        else:
            results.update({
                "avg_sv": 0.0, "avg_sm": 0.0, "avg_vc": 0.0, "avg_overall": 0.0,
                "std_sv": 0.0, "std_sm": 0.0, "std_vc": 0.0, "std_overall": 0.0
            })
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = "zero_shot_results") -> Path:
        """Save results to JSON file."""
        # Create output directory
        output_path = Path(output_dir) / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on dataset
        dataset_name = results["dataset"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zero_shot_{dataset_name}_{timestamp}.json"
        filepath = output_path / filename
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        results_converted = convert_types(results)
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"  Results saved to: {filepath}")
        return filepath
    
    def print_summary(self, results: Dict):
        """Print summary of evaluation results."""
        print(f"\n{'='*80}")
        print(f"ZERO-SHOT EVALUATION SUMMARY: {results['dataset']}")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {results['dataset']}")
        print(f"Examples: {results['total_examples']}")
        print(f"SV:  {results.get('avg_sv', 0):.4f} ± {results.get('std_sv', 0):.4f}")
        print(f"SM:  {results.get('avg_sm', 0):.4f} ± {results.get('std_sm', 0):.4f}")
        print(f"VC:  {results.get('avg_vc', 0):.4f} ± {results.get('std_vc', 0):.4f}")
        print(f"Overall: {results.get('avg_overall', 0):.4f} ± {results.get('std_overall', 0):.4f}")
        print(f"{'='*80}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation for DeepSeek-Coder-V2-Lite on OOD datasets"
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to DeepSeek-Coder-V2-Lite directory")
    parser.add_argument("--data_dir", type=str, default="data/unseen",
                       help="Directory containing OOD datasets")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum examples to evaluate per dataset (default: all)")
    parser.add_argument("--output_dir", type=str, default="zero_shot_deepseek_v2",
                       help="Output directory for results")
    parser.add_argument("--skip_metrics", action="store_true",
                       help="Skip metric computation (just generate text)")
    parser.add_argument("--datasets", type=str, default="all",
                       help="Comma-separated list of datasets to evaluate (lea,rectangle,xtea)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("DEEPSEEK-V2-LITE ZERO-SHOT EVALUATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max examples per dataset: {args.max_examples or 'all'}")
    print("="*80)
    
    # Determine which datasets to evaluate
    if args.datasets.lower() == "all":
        datasets_to_evaluate = ["lea", "rectangle", "xtea"]
    else:
        datasets_to_evaluate = [ds.strip().lower() for ds in args.datasets.split(",")]
    
    print(f"Datasets to evaluate: {datasets_to_evaluate}")
    
    # Initialize evaluator
    config = get_deepseek_v2_config()
    evaluator = DeepSeekV2ZeroShotEvaluator(args.model, config)
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets_to_evaluate:
        dataset_file = Path(args.data_dir) / f"{dataset_name}_dataset.jsonl"
        
        if not dataset_file.exists():
            print(f"\nWarning: Dataset file not found: {dataset_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"EVALUATING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = evaluator.load_dataset(str(dataset_file))
        
        if not dataset:
            print(f"Error: Could not load dataset from {dataset_file}")
            continue
        
        print(f"Loaded {len(dataset)} examples")
        
        # Evaluate
        results = evaluator.evaluate_dataset(
            dataset, 
            dataset_name, 
            max_examples=args.max_examples
        )
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Store for overall comparison
        all_results[dataset_name] = results
    
    # Generate overall comparison
    if len(all_results) > 0:
        generate_comparison_table(all_results, args.output_dir, evaluator.model_name)
    
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION COMPLETE!")
    print("="*80)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def generate_comparison_table(all_results: Dict[str, Dict], output_dir: str, model_name: str):
    """Generate comparison table across datasets."""
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    table_file = output_path / "zero_shot_comparison.txt"
    latex_file = output_path / "zero_shot_comparison.tex"
    
    # Text table
    with open(table_file, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"DEEPSEEK-V2-LITE ZERO-SHOT PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"{'Dataset':<15} {'Examples':<10} {'SV':<10} {'SM':<10} {'VC':<10} {'Overall':<10}\n")
        f.write("-"*80 + "\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"{dataset_name:<15} ")
            f.write(f"{results['total_examples']:<10} ")
            f.write(f"{results.get('avg_sv', 0):<10.3f} ")
            f.write(f"{results.get('avg_sm', 0):<10.3f} ")
            f.write(f"{results.get('avg_vc', 0):<10.3f} ")
            f.write(f"{results.get('avg_overall', 0):<10.3f}\n")
        
        f.write("="*80 + "\n")
        f.write("SV = Syntax Validity, SM = Semantic Match, VC = Value Consistency\n")
    
    print(f"\nComparison table saved to: {table_file}")
    
    # LaTeX table
    generate_latex_table(all_results, model_name, latex_file)

def generate_latex_table(all_results: Dict[str, Dict], model_name: str, output_file: Path):
    """Generate LaTeX table for paper."""
    content = f"""\\begin{{table}}[t]
\\caption{{Zero-shot Performance of {model_name} on OOD Cryptographic Translation}}
\\label{{tab:zero_shot_{model_name.lower().replace('-', '_')}}}
\\centering
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{SV}} & \\textbf{{SM}} & \\textbf{{VC}} & \\textbf{{Overall}} \\\\
\\midrule
"""
    
    for dataset_name, results in all_results.items():
        dataset_cap = dataset_name.capitalize()
        sv = results.get('avg_sv', 0.0)
        sm = results.get('avg_sm', 0.0)
        vc = results.get('avg_vc', 0.0)
        overall = results.get('avg_overall', 0.0)
        
        content += f"{dataset_cap} & {sv:.3f} & {sm:.3f} & {vc:.3f} & {overall:.3f} \\\\\n"
    
    content += """\\bottomrule
\\end{tabular}
\\vspace{0.2cm}
\\footnotesize
\\textit{Note: Zero-shot evaluation shows untrained model performance on unseen ciphers.}
\\end{table}"""
    
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"LaTeX table saved to: {output_file}")

# ============================================================================
if __name__ == "__main__":
    main()
