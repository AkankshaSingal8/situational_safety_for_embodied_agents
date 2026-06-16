#!/usr/bin/env python3
"""
merge_lora.py — Merge DreamZero-LIBERO-LoRA adapter into DreamZero-AgiBot base model.

Run once (takes ~10-20 min on CPU, or ~5 min on GPU).
The merged checkpoint is then passed to socket_test_optimized_AR.py via --model-path.

Usage:
    conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env
    python dream-policy/merge_lora.py

Output:
    /ocean/projects/cis250185p/asingal/.hf_cache/hub/DreamZero-LIBERO-merged/
"""

import os
import sys
from pathlib import Path

HF_CACHE = "/ocean/projects/cis250185p/asingal/.hf_cache"
BASE_PATH = f"{HF_CACHE}/hub/GEAR-Dreams/DreamZero-AgiBot"
LORA_PATH = f"{HF_CACHE}/hub/KyleZ0906/DreamZero-LIBERO-LoRA"
OUTPUT_PATH = f"{HF_CACHE}/hub/DreamZero-LIBERO-merged"

os.environ["HF_HOME"] = HF_CACHE

def main():
    print(f"Loading base model from: {BASE_PATH}")
    print(f"Applying LoRA from:      {LORA_PATH}")
    print(f"Saving merged model to:  {OUTPUT_PATH}")

    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoConfig
    except ImportError:
        print("ERROR: peft and transformers are required.")
        print("  conda activate /ocean/projects/cis250185p/asingal/envs/dreamzero_env")
        sys.exit(1)

    # Try standard transformers AutoModel loading
    # DreamZero may use a custom model class — adjust if this fails
    try:
        print("Loading base model weights...")
        config = AutoConfig.from_pretrained(BASE_PATH, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            BASE_PATH,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    except Exception as e:
        print(f"ERROR loading base model: {e}")
        print("The DreamZero model may require custom loading code from the dreamzero repo.")
        print("Try: pip install -e dream-policy/dreamzero && re-run with dreamzero's model class.")
        sys.exit(1)

    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH, trust_remote_code=True)

    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {OUTPUT_PATH} ...")
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUTPUT_PATH)

    # Copy tokenizer/processor files if present
    import shutil
    for fname in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                  "config.json", "generation_config.json"]:
        src = Path(BASE_PATH) / fname
        if src.exists():
            shutil.copy(src, Path(OUTPUT_PATH) / fname)
            print(f"  Copied {fname}")

    print(f"\nDone. Merged checkpoint saved to: {OUTPUT_PATH}")
    print("Pass this path to the server: --model-path", OUTPUT_PATH)


if __name__ == "__main__":
    main()
