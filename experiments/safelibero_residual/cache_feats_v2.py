"""Frozen SigLIP features for the v2 residual data (no fine-tuning; run once, train many times).

    python cache_feats_v2.py --suites spatial,object,goal,long --rounds r0 --out data_v2/feats_siglip

Per suite, from data_v2/<suite>/<round>/ep_*.npz:
  <out>/<suite>_agent.npy, <suite>_wrist.npy  (N,49,768) fp16   SigLIP-base patch tokens, 14x14 avg-pooled to 7x7
  <out>/<suite>_index.npz                     state (N,8), base (N,10,7), drive_tgt (N,10,3), ep (N,), chunk (N,)
                                              and eps_meta (json per episode: suite, scene, level, task, lang, drive, others)
Frames are stored in raw env orientation; they are rotated 180 deg (upright) before encoding, here and at eval.
Text: <out>/text_bank.npz holds SigLIP text embeddings (pooled, L2-normalised) of every safety sentence
(train + held-out wordings) and every task instruction seen in the data. The empty sentence is not embedded;
the model has its own learned null token.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

MODEL = "google/siglip-base-patch16-224"


def load_siglip(dev):
    from transformers import SiglipModel, AutoTokenizer
    m = SiglipModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    return m, AutoTokenizer.from_pretrained(MODEL)


@torch.no_grad()
def encode_images(model, imgs_u8, dev, bs=128):
    """imgs_u8: (N,224,224,3) raw env orientation -> (N,49,768) fp16 numpy."""
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs_u8[i:i + bs, ::-1, ::-1])).to(dev)
        x = x.permute(0, 3, 1, 2).half() / 255.0
        x = (x - 0.5) / 0.5
        h = model.vision_model(pixel_values=x).last_hidden_state          # B,196,768
        h = h.view(-1, 14, 14, 768).permute(0, 3, 1, 2)
        h = torch.nn.functional.avg_pool2d(h.float(), 2).flatten(2).transpose(1, 2)  # B,49,768
        out.append(h.half().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_texts(model, tok, texts, dev):
    b = tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
    e = model.get_text_features(**b).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class Eps(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        return {k: z[k] for k in ("agent", "wrist", "state", "base", "drive_tgt")} | {"meta": str(z["meta"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="spatial,object,goal,long")
    ap.add_argument("--rounds", default="r0", help="comma-separated round dirs under data_v2/<suite>/")
    ap.add_argument("--out", default="data_v2/feats_siglip")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    model, tok = load_siglip(dev)
    langs = set()
    for s in a.suites.split(","):
        files = sorted(sum((glob.glob(f"data_v2/{s}/{r}/ep_*.npz") for r in a.rounds.split(",")), []))
        dl = torch.utils.data.DataLoader(Eps(files), batch_size=None, num_workers=8, prefetch_factor=4)
        A, W, ST, B, T, EP, CH, M = [], [], [], [], [], [], [], []
        for e, z in enumerate(dl):
            meta = json.loads(z["meta"])
            n = len(z["state"])
            A.append(encode_images(model, z["agent"].numpy(), dev))
            W.append(encode_images(model, z["wrist"].numpy(), dev))
            ST.append(z["state"].numpy()); B.append(z["base"].numpy()); T.append(z["drive_tgt"].numpy())
            EP.append(np.full(n, e, np.int32)); CH.append(np.arange(n, dtype=np.int32))
            M.append(json.dumps({k: meta[k] for k in ("scene", "layout", "level", "task", "lang", "drive", "others")}
                                | {"suite": s, "file": os.path.relpath(files[e])}))
            langs.add(meta["lang"])
            if e % 200 == 0:
                print(f"{s}: {e}/{len(files)} eps", flush=True)
        np.save(f"{a.out}/{s}_agent.npy", np.concatenate(A)); np.save(f"{a.out}/{s}_wrist.npy", np.concatenate(W))
        np.savez(f"{a.out}/{s}_index.npz", state=np.concatenate(ST), base=np.concatenate(B),
                 drive_tgt=np.concatenate(T), ep=np.concatenate(EP), chunk=np.concatenate(CH),
                 eps_meta=np.array(M))
        print(f"{s}: {len(files)} eps, {sum(len(x) for x in ST)} chunks", flush=True)
        del A, W
    import texts as TX
    sents = sorted({t for o in TX.OBJ for t in TX.train_texts(o) + TX.held_texts(o)})
    old = {}
    if os.path.exists(f"{a.out}/text_bank.npz"):
        z = np.load(f"{a.out}/text_bank.npz", allow_pickle=True)
        old = dict(zip(z["texts"].tolist(), z["emb"]))
    allt = sorted(set(sents) | langs | set(old))
    emb = encode_texts(model, tok, allt, dev)
    np.savez(f"{a.out}/text_bank.npz", texts=np.array(allt, dtype=object), emb=emb)
    print("text bank", len(allt), flush=True)


if __name__ == "__main__":
    main()
