#!/usr/bin/env python3
"""
Evaluate baseline (untrained) models on cryptographic translation.
Automatically discovers models in models/ directory.
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
from typing import Dict, List, Optional, Tuple
import os

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# --- DEEPSEEK FLASH_ATTN WORKAROUND ---
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Removes flash_attn from the required imports list to bypass the error."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

#from transformers.cache_utils import DynamicCache

# globally
# Comprehensive Monkeypatch for DeepSeek-V2-Lite + Transformers 4.57+
#def apply_deepseek_cache_patch():
#    # 1. Patch 'seen_tokens' as a property
#    if not hasattr(DynamicCache, "seen_tokens"):
#        @property
#        def seen_tokens(self):
#            return self.get_seq_length()
#        DynamicCache.seen_tokens = seen_tokens

#    # 2. Patch 'get_max_length' 
#    if not hasattr(DynamicCache, "get_max_length"):
#        def get_max_length(self):
#            # Return a large number if not specified to avoid truncation logic errors
#            return getattr(self, "max_cache_size", 4096) 
#        DynamicCache.get_max_length = get_max_length

#    # 3. Some versions look for the underscore version internally
#    if not hasattr(DynamicCache, "_seen_tokens"):
#        @property
#        def _seen_tokens(self):
#            return self.get_seq_length()
#        DynamicCache._seen_tokens = _seen_token

# Import metrics
try:
    from evaluation.metrics_fixed import FixedCryptographicMetrics
    print("✓ Using FixedCryptographicMetrics")
    METRICS_CLASS = FixedCryptographicMetrics
except ImportError:
    print("✗ FixedCryptographicMetrics not found, using basic evaluation")
    METRICS_CLASS = None


class BaselineModelEvaluator:
    """Evaluates untrained baseline models."""
    
    def __init__(self, model_path: str, model_name: str = None, 
                 use_4bit: bool = False, max_length: int = 512, output_dir='zero_shot_evaluation'):
        """
        Args:
            model_path: Path to model directory
            model_name: Optional custom name (defaults to directory name)
            use_4bit: Use 4-bit quantization
            max_length: Maximum sequence length
        """
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.name
        self.max_length = max_length
        self.use_4bit = use_4bit
        
        # Output directory
        self.output_dir = Path(output_dir) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading baseline model from: {model_path}")
        print(f"Model name: {self.model_name}")
        
        # Try to determine model size from name
        self.model_size = self._infer_model_size()
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except:
            print(f"Warning: Could not load tokenizer from {self.model_path}")
            print("Trying with trust_remote_code=True...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if model is sharded
        is_sharded = self._is_model_sharded()
        

        # Use the patch to bypass the flash_attn check during model loading
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): 
            # Load with appropriate precision
            if use_4bit:
                print("Using 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    quantization_config=bnb_config,
                    device_map="auto", # "cuda" for deepseek lite "auto" for the other models
                    trust_remote_code=True,
                    use_cache=True,
                    #attn_implementation="eager",
                 )
            else:
                print("Using FP16/BF16 precision...")
            
                # Check model config for dtype
                config_path = self.model_path / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        torch_dtype = config.get("torch_dtype", "float16")
                        if torch_dtype == "bfloat16":
                            torch_dtype = torch.bfloat16
                        else:
                            torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float16



                if "v2" in model_name.lower():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        torch_dtype=torch_dtype,
                        #device_map="auto",
                        trust_remote_code=True,
                        use_cache=True,
                        #max_position_embeddings=4096,  # Reduce from 128K
                        #sliding_window=1024,  # Optional: limit sliding window attention
                        attn_implementation="eager"
                    )
                    # Modify the config after loading
                    # Force rope scaling to use smaller context
                    if hasattr(self.model.config, 'rope_scaling'):
                        print(f"Original rope_scaling: {self.model.config.rope_scaling}")
                        
                        # Force YARN scaling to use smaller context
                        self.model.config.rope_scaling = {
                            "type": "yarn",
                            "factor": 1.0,  # No scaling
                            "original_max_position_embeddings": 4096,
                            "max_position_embeddings": 4096,
                        }
                        print(f"Modified rope_scaling: {self.model.config.rope_scaling}")
                    
                    # Also set max_position_embeddings
                    original_max = self.model.config.max_position_embeddings
                    self.model.config.max_position_embeddings = 4096
                    print(f"Reduced max_position_embeddings from {original_max} to 4096")

                    #original_max_pos = self.model.config.max_position_embeddings
                    #self.model.config.max_position_embeddings = 4096
                    #print(f"Reduced max_position_embeddings from {original_max_pos} to 4096")
                    

                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        torch_dtype=torch_dtype,
                        device_map="auto", #"cuda", # "cuda" for deepseek lite "auto" for the other models
                        trust_remote_code=True,
                        use_cache=False, 
                    )
            
               
        
        # Initialize metrics
        if METRICS_CLASS:
            self.metrics = METRICS_CLASS()
        else:
            self.metrics = None
        
        print(f"✓ Baseline model loaded: {self.model_name} ({self.model_size})")
    
    def _infer_model_size(self) -> str:
        """Infer model size from directory name."""
        name = self.model_path.name.lower()
        
        # Common patterns
        patterns = [
            (r"1\.3b|1.3b", "1.3B"),
            (r"3b|3B", "3B"),
            (r"6\.7b|6.7b", "6.7B"),
            (r"7b|7B", "7B"),
            (r"13b|13B", "13B"),
            (r"33b|33B", "33B"),
            (r"lite", "Lite"),  # DeepSeek-V2-Lite
        ]
        
        for pattern, size in patterns:
            if re.search(pattern, name):
                return size
        
        # Check config file
        config_path = self.model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    # Try to infer from hidden_size and num_layers
                    hidden_size = config.get("hidden_size", 0)
                    num_layers = config.get("num_hidden_layers", 0)
                    
                    if hidden_size > 4000:
                        return "Large"
                    elif hidden_size > 2000:
                        return "Medium"
                    else:
                        return "Small"
            except:
                pass
        
        return "Unknown"
    
    def _is_model_sharded(self) -> bool:
        """Check if model is sharded (split into multiple files)."""
        # Check for safetensors index
        if (self.model_path / "model.safetensors.index.json").exists():
            return True
        
        # Check for pytorch bin index
        if (self.model_path / "pytorch_model.bin.index.json").exists():
            return True
        
        # Check for multiple safetensors files
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        if len(safetensors_files) > 1:
            return True
        
        # Check for multiple bin files
        bin_files = list(self.model_path.glob("*.bin"))
        if len(bin_files) > 1:
            return True
        
        return False
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load JSONL dataset."""
        with open(filepath, "r") as f:
            return [json.loads(line) for line in f]
    
    def extract_family_from_metadata(self, example: Dict) -> str:
        """Extract family from example metadata."""
        metadata = example.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("family", "unknown")
        return "unknown"
    
    def generate_baseline_response(self, example: Dict) -> str:
        """
        Generate response from baseline model using zero-shot prompting.
        Uses deterministic greedy decoding for consistent results.
        """
        instruction = example.get("instruction", "")
        input_code = example.get("input", "")
        
        # Zero-shot prompt format
        prompt = f"### Instruction: {instruction}\n### Input: {input_code}\n### Output:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        
        
        if "v2" in self.model_name.lower():
            max_new_tokens = 384
            # Use this instead:
            #device = next(self.model.parameters()).device
            #inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        elif "starcoder" in self.model_name.lower():
            max_new_tokens = 768
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:             
            max_new_tokens = 512
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}


        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  #  SAME as training and evaluation
                do_sample=False,     # Deterministic
                use_cache=True,# CHANGE 1: Explicitly set use_cache to True and past_key_values to None
                #past_key_values=None, 
                # CHANGE 2: Force the use of the tuple-based cache 
                #cache_position=None, 
                # CHANGE 3: Keep this to ensure internal HF logic remains legacy-friendly
                #return_legacy_cache=True,
                temperature=None,
                #attn_implementation="eager",  # This bypasses the flash_attn check
                top_p=None,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
                
                
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part
        if "### Output:" in full_text:
            generated = full_text.split("### Output:")[-1].strip()
        else:
            generated = full_text
        
        # Clean up
        generated = self._clean_generated_text(generated)
        return generated
    
    def _clean_generated_text(self, generated: str) -> str:
        """Clean up generated text."""
        # Stop patterns
        stop_patterns = [
            "\n###",
            "\n\nend",
            "\n(*",
            "\n\ndefinition ",
            "\n\nfun ",
            "\nNote:",
            "\nwhere\n  ",
        ]
        
        # Find earliest stop
        earliest = len(generated)
        for pattern in stop_patterns:
            idx = generated.find(pattern)
            if 0 <= idx < earliest:
                earliest = idx
        
        if earliest < len(generated):
            generated = generated[:earliest].strip()
        
        return generated
    
    def compute_metrics_single(self, generated: str, reference: str) -> Dict:
        """Compute SV, SM, VC metrics."""
        if not self.metrics:
            return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}
        
        try:
            results = self.metrics.evaluate(generated, reference)
            
            sv = results.get("syntax_validity", 0.0)
            sm = results.get("semantic_match", 0.0)
            vc = results.get("value_consistency", 0.0)
            
            return {
                "sv": float(sv),
                "sm": float(sm),
                "vc": float(vc),
                "overall": float((sv + sm + vc) / 3.0)
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}
    
    def evaluate_test_set(self, test_data: List[Dict], max_examples: Optional[int] = None) -> Dict:
        """Evaluate on test set."""
        if max_examples:
            test_data = test_data[:max_examples]
        
        results = {
            "model": self.model_name,
            "model_path": str(self.model_path),
            "model_size": self.model_size,
            "status": "baseline_untrained",
            "total_examples": len(test_data),
            "sv_scores": [],
            "sm_scores": [],
            "vc_scores": [],
            "overall_scores": [],
            "examples": [],
            "family_stats": {},
            "timestamp": datetime.now().isoformat(),
            "use_4bit": self.use_4bit,
            "max_length": self.max_length,
        }
        
        print(f"\nEvaluating baseline {self.model_name} on {len(test_data)} examples...")
        
        for i, example in enumerate(tqdm(test_data, desc=f"Baseline {self.model_name}")):
            # Generate from baseline model
            generated = self.generate_baseline_response(example)
            generated = clean_code(generated)
            reference = example.get("output", "")
            family = self.extract_family_from_metadata(example)
            
            # Compute metrics
            metrics = self.compute_metrics_single(generated, reference)
            
            # Store results
            results["sv_scores"].append(metrics["sv"])
            results["sm_scores"].append(metrics["sm"])
            results["vc_scores"].append(metrics["vc"])
            results["overall_scores"].append(metrics["overall"])
            
            # Track by family
            if family not in results["family_stats"]:
                results["family_stats"][family] = {
                    "sv": [], "sm": [], "vc": [], "overall": [], "count": 0
                }
            
            results["family_stats"][family]["sv"].append(metrics["sv"])
            results["family_stats"][family]["sm"].append(metrics["sm"])
            results["family_stats"][family]["vc"].append(metrics["vc"])
            results["family_stats"][family]["overall"].append(metrics["overall"])
            results["family_stats"][family]["count"] += 1
            
            # Store example (truncated)
            results["examples"].append({
                "index": i,
                "family": family,
                "generated": generated[:300] + "..." if len(generated) > 300 else generated,
                "generated_full": generated if len(generated) <= 500 else None,
                "reference": reference[:300] + "..." if len(reference) > 300 else reference,
                "reference_full": reference if len(reference) <= 500 else None,
                "sv": metrics["sv"],
                "sm": metrics["sm"],
                "vc": metrics["vc"],
                "overall": metrics["overall"]
            })
        
        # Calculate statistics
        self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: Dict) -> None:
        """Calculate statistics."""
        # Overall averages
        results["avg_sv"] = np.mean(results["sv_scores"]) if results["sv_scores"] else 0.0
        results["avg_sm"] = np.mean(results["sm_scores"]) if results["sm_scores"] else 0.0
        results["avg_vc"] = np.mean(results["vc_scores"]) if results["vc_scores"] else 0.0
        results["avg_overall"] = np.mean(results["overall_scores"]) if results["overall_scores"] else 0.0
        
        # Standard deviations
        results["std_sv"] = np.std(results["sv_scores"]) if results["sv_scores"] else 0.0
        results["std_sm"] = np.std(results["sm_scores"]) if results["sm_scores"] else 0.0
        results["std_vc"] = np.std(results["vc_scores"]) if results["vc_scores"] else 0.0
        results["std_overall"] = np.std(results["overall_scores"]) if results["overall_scores"] else 0.0
        
        # Family averages
        for family in results["family_stats"]:
            stats = results["family_stats"][family]
            stats["avg_sv"] = np.mean(stats["sv"]) if stats["sv"] else 0.0
            stats["avg_sm"] = np.mean(stats["sm"]) if stats["sm"] else 0.0
            stats["avg_vc"] = np.mean(stats["vc"]) if stats["vc"] else 0.0
            stats["avg_overall"] = np.mean(stats["overall"]) if stats["overall"] else 0.0
    
    def save_results(self, results: Dict, test_file: str ) -> Path:
        """Save results to file."""
        # Extract just the filename without extension (e.g., 'lea_dataset')
        test_filename = Path(test_file).stem
        output_file = self.output_dir / f"baseline_results_{test_filename}.json"
        
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
        
        results_converted = convert_numpy(results)
        
        with open(output_file, "w") as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved to: {output_file}")
        return output_file
    
    def generate_summary(self, results: Dict) -> str:
        """Generate summary report."""
        summary = []
        summary.append("="*80)
        summary.append(f"BASELINE EVALUATION: {self.model_name}")
        summary.append("="*80)
        summary.append(f"Model: {self.model_name} ({self.model_size})")
        summary.append(f"Path: {self.model_path}")
        summary.append(f"Status: Untrained baseline")
        summary.append(f"Samples: {results['total_examples']}")
        summary.append(f"SV: {results.get('avg_sv', 0):.4f} ± {results.get('std_sv', 0):.4f}")
        summary.append(f"SM: {results.get('avg_sm', 0):.4f} ± {results.get('std_sm', 0):.4f}")
        summary.append(f"VC: {results.get('avg_vc', 0):.4f} ± {results.get('std_vc', 0):.4f}")
        summary.append(f"Overall: {results.get('avg_overall', 0):.4f} ± {results.get('std_overall', 0):.4f}")
        
        # Per-family breakdown
        if results.get("family_stats"):
            summary.append("\nBy cipher family:")
            for family, stats in results["family_stats"].items():
                if stats["count"] > 0:
                    summary.append(f"  {family}: SM={stats.get('avg_sm', 0):.3f}, n={stats['count']}")
                    summary.append(f"  {family}: SV={stats.get('avg_sv', 0):.3f}, n={stats['count']}")
                    summary.append(f"  {family}: VC={stats.get('avg_vc', 0):.3f}, n={stats['count']}")
        
        summary.append("="*80)
        
        return "\n".join(summary)


def discover_models(models_dir: str = "models") -> List[Tuple[str, str]]:
    """
    Discover all models in the models directory.
    Returns list of (model_path, model_name) tuples.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return []
    
    models = []
    
    for item in models_dir.iterdir():
        if item.is_dir():
            # Check if this looks like a model directory
            model_files = [
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "model.safetensors.index.json",
            ]
            
            has_model_files = any((item / f).exists() for f in model_files)
            
            if has_model_files:
                models.append((str(item), item.name))
                print(f"  Found: {item.name}")
    
    # Sort by size (smallest first)
    def get_size_rank(name):
        name_lower = name.lower()
        if "1.3b" in name_lower: return 1
        if "3b" in name_lower: return 2
        if "6.7b" in name_lower: return 3
        if "7b" in name_lower: return 4
        if "13b" in name_lower: return 5
        if "33b" in name_lower: return 6
        return 99
    
    models.sort(key=lambda x: get_size_rank(x[1]))
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline (untrained) models")
    parser.add_argument("--models_dir", type=str, default="models",
                       help="Directory containing models")
    parser.add_argument("--test_file", type=str, default="data/unseen/",
                       help="Test dataset file")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to evaluate per model")
    parser.add_argument("--use_4bit", choices=["auto", "true", "false", "--use_4bit"], default="auto",
                        help="Whether to use 4-bit quantization" )
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Evaluate specific model (name in models directory)")
    parser.add_argument("--skip_large", action="store_true",
                       help="Skip models larger than 7B (for memory constraints)")
    parser.add_argument("--output_dir", type=str, default="zero_shot_evaluation",
                       help="Skip models larger than 7B (for memory constraints)")
    
    
    args = parser.parse_args()
    
    # Normalize use_4bit to boolean
    if args.use_4bit == "true" or args.use_4bit == "--use_4bit":
        use_4bit = True
    elif args.use_4bit == "false":
        use_4bit = False
    else:
        use_4bit = "auto"  # keep sentinel
    
    
    print("="*80)
    print("BASELINE MODEL EVALUATION FOR ICML PAPER")
    print("="*80)
    print(f"Models directory: {args.models_dir}")
    print(f"Test file: {args.test_file}")
    print(f"Max samples: {args.max_samples}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"Skip large models (>7B): {args.skip_large}")
    print("="*80)
    
    # Load test data
    test_data = []
    try:
        with open(args.test_file, "r") as f:
            test_data = [json.loads(line) for line in f]
        print(f"Loaded {len(test_data)} test examples")
    except FileNotFoundError:
        print(f"Error: Test file not found: {args.test_file}")
        return
    
    # Discover models
    print("\nDiscovering models...")
    all_models = discover_models(args.models_dir)
    
    if not all_models:
        print(f"No models found in {args.models_dir}")
        return
    
    print(f"Found {len(all_models)} models")
    
    all_results = {}
    
    # Evaluate specific model or all discovered models
    if args.model_name:
        # Find the specific model
        found = False
        for model_path, model_name in all_models:
            if model_name == args.model_name:
                print(f"\nEvaluating specific model: {model_name}")
                
                # Determine if we should use 4-bit for this model
                model_use_4bit = use_4bit

                if model_use_4bit == "auto":
                    model_use_4bit = (
                        ("6.7b" in model_name.lower()
                        or "7b" in model_name.lower()
                        #or "13b" in model_name.lower()
                        #or "33b" in model_name.lower()
                        ) and "v2" in model_name.lower()
                    )
                
                evaluator = BaselineModelEvaluator(
                    model_path=model_path,
                    model_name=model_name,
                    use_4bit=model_use_4bit,
                    max_length=args.max_length,
                    output_dir = args.output_dir
                )
                lengths = [
                    len(evaluator.tokenizer(example["output"]).input_ids)
                    for example in test_data[:args.max_samples]
                ]
                
                
                print("\n=== Reference Output Token Length Stats ===")
                print(f"Max tokens Input  : {args.max_length}")
                print(f"Max tokens        : {max(lengths)}")
                print(f"95th percentile   : {np.percentile(lengths, 95):.1f}")
                print(f"90th percentile   : {np.percentile(lengths, 90):.1f}")
                print(f"Median            : {np.median(lengths):.1f}")
                print("===========================================\n")
                
                
                results = evaluator.evaluate_test_set(test_data[:args.max_samples])
                evaluator.save_results(results, args.test_file)
                
                print("\n" + evaluator.generate_summary(results))
                
                all_results[model_name] = results
                found = True
                break
        
        if not found:
            print(f"Model '{args.model_name}' not found in {args.models_dir}")
            print(f"Available models: {[name for _, name in all_models]}")
    
    else:
        # Evaluate all discovered models
        for i, (model_path, model_name) in enumerate(all_models):
            print(f"\n{'='*60}")
            print(f"MODEL {i+1}/{len(all_models)}: {model_name}")
            print(f"{'='*60}")
            
            # Skip large models if requested
            if args.skip_large:
                if any(size in model_name.lower() for size in ["13b", "33b", "large"]):
                    print(f"Skipping large model: {model_name}")
                    continue
            
            # Determine if we should use 4-bit for this model
            model_use_4bit = use_4bit
            if isinstance(model_use_4bit, str) and model_use_4bit == "auto":
                model_use_4bit = any(size in model_name.lower() 
                                   for size in ["6.7b", "13b", "33b"])
            
            try:
                evaluator = BaselineModelEvaluator(
                    model_path=model_path,
                    model_name=model_name,
                    use_4bit=model_use_4bit,
                    max_length=args.max_length,
                    output_dir = args.output_dir
                
                )
                
                results = evaluator.evaluate_test_set(test_data[:args.max_samples])
                evaluator.save_results(results, args.test_file)
                
                print("\n" + evaluator.generate_summary(results))
                
                all_results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                print("Skipping to next model...")
    
    # Generate comparison table
    print ('if len(all_results) > 1: ', len(all_results) > 1)
    if len(all_results) >= 1:
        output_path = args.output_dir + '/' + args.model_name
        generate_comparison_table(all_results, output_path ,  args.test_file)
        #generate_comparison_table(all_results, args.output_dir,  args.test_file)
    
    print("\n" + "="*80)
    print("BASELINE EVALUATION COMPLETE!")
    print("="*80)




def clean_code(code_str: str) -> str:
    # Remove single-line comments
    code_str = re.sub(r'#.*', '', code_str)
    # Remove multi-line docstrings
    code_str = re.sub(r'"""(.*?)"""', '', code_str, flags=re.DOTALL)
    # Strip trailing whitespace and blank lines
    code_str = "\n".join(line.rstrip() for line in code_str.splitlines() if line.strip())
    return code_str


def generate_comparison_table(all_results: Dict[str, Dict], output_dir:str, test_file:str):
    """Generate comparison table of all baseline models."""
    
    # Extract just the filename without extension (e.g., 'lea_dataset')
    test_filename = Path(test_file).stem
    table_file =  output_dir + f"/baseline_comparison_{test_filename}.txt"
    # Path("results/baselines/baseline_comparison.txt")
    Path(table_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(table_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODEL COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<30} {'Size':<8} {'SV':<8} {'SM':<8} {'VC':<8} {'Overall':<8} {'Samples':<8}\n")
        f.write("-"*80 + "\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name:<30} ")
            f.write(f"{results.get('model_size', 'N/A'):<8} ")
            f.write(f"{results.get('avg_sv', 0):<8.3f} ")
            f.write(f"{results.get('avg_sm', 0):<8.3f} ")
            f.write(f"{results.get('avg_vc', 0):<8.3f} ")
            f.write(f"{results.get('avg_overall', 0):<8.3f} ")
            f.write(f"{results.get('total_examples', 0):<8}\n")
        
        f.write("="*80 + "\n")
        f.write("Note: SV=Syntax Validity, SM=Semantic Match, VC=Value Consistency\n")
    
    print(f"\nComparison table saved to: {table_file}")
    
    # Also generate LaTeX table for paper
    generate_latex_table(all_results, output_dir, test_file)


def generate_latex_table(all_results: Dict[str, Dict], output_dir:str, test_file:str):
    """Generate LaTeX table for paper."""
    
    # Extract just the filename without extension (e.g., 'lea_dataset')
    test_filename = Path(test_file).stem
    
    latex_file = output_dir + f"/baseline_comparison_{test_filename}.tex"
    # Path("results/baselines/baseline_comparison.tex")
    Path(latex_file).parent.mkdir(parents=True, exist_ok=True)
    
    content = """\\begin{table}[t]
\\caption{Baseline Performance of Untrained Models on Cryptographic Translation}
\\label{tab:baseline_results}
\\centering
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{SV} & \\textbf{SM} & \\textbf{VC} & \\textbf{Overall} \\\\
\\midrule
"""
    
    for model_name, results in sorted(all_results.items()):
        # Shorten model name for table
        short_name = model_name
        # Remove common prefixes
        short_name = short_name.replace("deepseek-coder-", "DeepSeek-")
        short_name = short_name.replace("starcoder2-", "StarCoder2-")
        short_name = short_name.replace("DeepSeek-Coder-V2-Lite", "DeepSeek-V2-Lite")
        short_name = short_name.replace("-instruct", "")
        short_name = short_name.replace("-base", "")
        
        sv = results.get('avg_sv', 0.0)
        sm = results.get('avg_sm', 0.0)
        vc = results.get('avg_vc', 0.0)
        overall = results.get('avg_overall', 0.0)
        
        content += f"{short_name} & {sv:.3f} & {sm:.3f} & {vc:.3f} & {overall:.3f} \\\\\n"
    
    content += """\\bottomrule
\\end{tabular}
\\vspace{0.2cm}
\\footnotesize
\\textit{Note: Baseline evaluation shows untrained models struggle with cryptographic translation (low SM scores).}
\\end{table}"""
    
    with open(latex_file, "w") as f:
        f.write(content)
    
    print(f"LaTeX table saved to: {latex_file}")


if __name__ == "__main__":
    main()
