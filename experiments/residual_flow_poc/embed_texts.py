"""Encode every instruction once with frozen BGE-base (CLS pooling, L2 norm).

Runs in the `libero` env (transformers 4.21); the flow model only ever sees the
saved vectors. The NULL instruction maps to the zero vector (the model has a
learned null token instead).
    python embed_texts.py --out data/text_emb.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.texts import all_texts, NULL  # noqa: E402

MODEL = os.environ.get("BGE_PATH", "BAAI/bge-base-en-v1.5")  # snapshot dir for old transformers

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/text_emb.npz")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).eval()
texts = all_texts()
real = [t for t in texts if t != NULL]
with torch.no_grad():
    b = tok(real, padding=True, truncation=True, return_tensors="pt")
    cls = enc(**b).last_hidden_state[:, 0]
    cls = torch.nn.functional.normalize(cls, dim=-1).numpy()
emb = {t: v for t, v in zip(real, cls)}
emb[NULL] = np.zeros(cls.shape[1], dtype=np.float32)
np.savez(args.out, texts=np.array(texts, dtype=object), emb=np.stack([emb[t] for t in texts]))
print(f"saved {len(texts)} embeddings dim={cls.shape[1]} -> {args.out}")
