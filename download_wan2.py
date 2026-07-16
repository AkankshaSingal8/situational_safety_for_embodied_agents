"""
Pre-download Wan2.1-I2V-14B-480P components that DreamZero's load_lora() needs.
Mirrors exactly what wan_flow_matching_action_tf.py fetches at runtime.
"""
import os, json

os.environ["HF_HOME"] = "/ocean/projects/cis250185p/asingal/.hf_cache"

from huggingface_hub import hf_hub_download

REPO = "Wan-AI/Wan2.1-I2V-14B-480P"

print("=== Downloading Wan2.1-I2V-14B-480P components ===\n")

# 1. T5 text encoder (~10 GB)
print("[1/3] Text encoder: models_t5_umt5-xxl-enc-bf16.pth")
p = hf_hub_download(repo_id=REPO, filename="models_t5_umt5-xxl-enc-bf16.pth")
print(f"  -> {p}\n")

# 2. CLIP image encoder (~2 GB)
print("[2/3] Image encoder: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
p = hf_hub_download(repo_id=REPO, filename="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
print(f"  -> {p}\n")

# 3. VAE (~0.5 GB)
print("[3/3] VAE: Wan2.1_VAE.pth")
p = hf_hub_download(repo_id=REPO, filename="Wan2.1_VAE.pth")
print(f"  -> {p}\n")

# 4. DiT index + all 7 shards (~28 GB)
print("[4/4] DiT index + shards (this is the big one ~28 GB)")
index_path = hf_hub_download(repo_id=REPO, filename="diffusion_pytorch_model.safetensors.index.json")
print(f"  index -> {index_path}")
with open(index_path) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
print(f"  {len(shards)} shards to download:")
for i, shard in enumerate(shards, 1):
    print(f"  [{i}/{len(shards)}] {shard}")
    p = hf_hub_download(repo_id=REPO, filename=shard)
    size_gb = os.path.getsize(p) / 1e9
    print(f"    -> {p}  ({size_gb:.2f} GB)")

print("\n=== All downloads complete ===")
