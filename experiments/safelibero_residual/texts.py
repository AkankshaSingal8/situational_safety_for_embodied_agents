"""Instruction templates for "don't hit <object>" and the embedding cache.

Training wordings and nouns are disjoint from the held-out ones used in Exp 2-style tests.
The null instruction (no safety sentence) maps to the zero vector.
    python texts.py     # writes data/text_emb.npz (BGE-base CLS, L2-normalised)
"""
import os

import numpy as np

OBJ = ["moka_pot_obstacle_1", "white_storage_box_obstacle_1", "milk_obstacle_1",
       "wine_bottle_obstacle_1", "red_coffee_mug_obstacle_1", "yellow_book_obstacle_1"]
NOUN_TRAIN = {"moka_pot_obstacle_1": "moka pot", "white_storage_box_obstacle_1": "white storage box",
              "milk_obstacle_1": "milk carton", "wine_bottle_obstacle_1": "wine bottle",
              "red_coffee_mug_obstacle_1": "red coffee mug", "yellow_book_obstacle_1": "yellow book"}
NOUN_HELD = {"moka_pot_obstacle_1": "stovetop espresso maker", "white_storage_box_obstacle_1": "white bin",
             "milk_obstacle_1": "carton of milk", "wine_bottle_obstacle_1": "bottle of wine",
             "red_coffee_mug_obstacle_1": "red cup", "yellow_book_obstacle_1": "yellow hardcover"}
T_TRAIN = ["don't hit the {o}", "avoid the {o}", "do not touch the {o}", "stay away from the {o}",
           "be careful not to bump into the {o}", "keep clear of the {o}", "watch out for the {o}",
           "don't knock into the {o}"]
T_HELD = ["make sure you never collide with the {o}", "give the {o} plenty of room",
          "steer around the {o}", "do not disturb the {o}"]
NULL = ""
HERE = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.path.join(HERE, "data", "text_emb.npz")


def train_texts(obj):
    return [t.format(o=NOUN_TRAIN[obj]) for t in T_TRAIN]


def held_texts(obj):
    """Held-out wording x {train noun, held-out noun}, plus train wording x held-out noun."""
    out = [t.format(o=NOUN_TRAIN[obj]) for t in T_HELD] + [t.format(o=NOUN_HELD[obj]) for t in T_HELD]
    out += [t.format(o=NOUN_HELD[obj]) for t in T_TRAIN[:2]]
    return out


def all_texts():
    s = [NULL]
    for o in OBJ:
        s += train_texts(o) + held_texts(o)
    return s


class TextBank:
    def __init__(self, path=EMB_PATH):
        z = np.load(path, allow_pickle=True)
        self.idx = {t: i for i, t in enumerate(z["texts"].tolist())}
        self.emb = z["emb"].astype(np.float32)

    def __call__(self, text):
        return self.emb[self.idx[text]]


if __name__ == "__main__":
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-base-en-v1.5"
    tok, enc = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    texts = all_texts()
    real = texts[1:]
    with torch.no_grad():
        b = tok(real, padding=True, truncation=True, return_tensors="pt")
        cls = torch.nn.functional.normalize(enc(**b).last_hidden_state[:, 0], dim=-1).numpy()
    emb = np.vstack([np.zeros((1, cls.shape[1]), np.float32), cls])
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.savez(EMB_PATH, texts=np.array(texts, dtype=object), emb=emb)
    print("saved", len(texts), "embeddings", emb.shape)
