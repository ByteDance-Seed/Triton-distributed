#!/usr/bin/env python3
"""
Cache Hugging Face model configurations in simple directory structure
Usage: python cache_hf_configs.py
"""

import json
from pathlib import Path
from transformers import AutoConfig

# Model list
MODELS = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B",
    "ByteDance-Seed/Seed-OSS-36B-Instruct"
]

# Simple directory structure
CACHE_DIR = Path(".codebase/hf_configs")


def cache_model_config(model_name: str):
    """Cache model config in simple directory structure"""
    try:
        print(f"Caching {model_name}...")

        # Create simple directory structure
        model_dir = CACHE_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get and save config and tokenizer
        config = AutoConfig.from_pretrained(model_name)
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Cache tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_dir))

        print(f"✅ Cached: {config_path} + tokenizer")
        return True

    except Exception as e:
        print(f"❌ Failed to cache {model_name}: {e}")
        return False


def main():
    """Main function"""
    print("Starting simple cache structure...")

    success_count = 0
    for model_name in MODELS:
        if cache_model_config(model_name):
            success_count += 1

    print(
        f"\nCaching complete: {success_count}/{len(MODELS)} model configurations cached"
    )

    if success_count == len(MODELS):
        print("🎉 All model configurations cached successfully!")
    else:
        print(
            "⚠️  Some model configurations failed to cache, check network connection"
        )


if __name__ == "__main__":
    main()
