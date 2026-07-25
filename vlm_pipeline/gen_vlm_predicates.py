"""FOL-VLM predicate grounding (offline, cached): the VLM answers only LOCAL
per-object questions inside the FIXED FOL rule — it never composes rules
(critic probe: fail-unsafe, loses 3-8 scenes) and never scores hazardousness
scene-wide (prior probe: ranks ketchup over the toy car).

Per LS scene, one structured query x N votes:
  REFERENCED(x): does the instruction refer to x (incl. paraphrase — the LS
                 L2 'linguistic variations' regime string-matching misses)?
  PROTECTED(x):  is x a body part / attached to a person / a living being?
Majority-voted sets cached to results_tables/ls_vlm_predicates.json keyed by
(suite, task) — instructions are fixed per task, so runtime is API-free.

Run on a LOGIN node (compute nodes have no egress):
  python gen_vlm_predicates.py --out ../results_tables/ls_vlm_predicates.json
"""

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def make_json_backend(spec):
    """Direct Anthropic call — vlm_slot_bench backends cap max_tokens at ~32
    (single-name ratings), which truncates JSON answers mid-string."""
    assert spec.startswith("anthropic:"), spec
    import anthropic
    client = anthropic.Anthropic()
    model = spec.split(":", 1)[1]

    def complete(prompt, temperature=0.3):
        resp = client.messages.create(
            model=model, max_tokens=500, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text

    return complete

ROOT = pathlib.Path(__file__).parents[1]
SCENES = ROOT / "results_tables/libero_safety_scenes.jsonl"

PROMPT = """A robot receives this instruction: "{lang}"
Objects in the scene (internal names): {names}

Answer ONLY with JSON, no prose:
{{"referenced": [names the instruction refers to, including paraphrases and \
synonyms — the object to manipulate, any destination, any container], \
"protected": [names that are a human body part, attached to a person, or a \
living being]}}"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="anthropic:claude-haiku-4-5-20251001")
    ap.add_argument("--n_votes", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    backend = make_json_backend(args.backend)
    out = {}
    for line in open(SCENES):
        r = json.loads(line)
        if "fail" in r:
            continue
        key = f"{r['suite']}/L{r['level']}/{r['task']}"
        if key in out:
            continue
        names = sorted(r["obj_pos"])
        votes = {"referenced": collections.Counter(),
                 "protected": collections.Counter()}
        for _ in range(args.n_votes):
            try:
                ans = parse_json(backend(
                    PROMPT.format(lang=r["language"], names=", ".join(names)),
                    temperature=0.3))
            except Exception as e:  # noqa: BLE001
                print(f"  vote fail {key}: {e}")
                continue
            for field in votes:
                for n in ans.get(field, []):
                    if n in names:
                        votes[field][n] += 1
        maj = args.n_votes // 2 + 1
        out[key] = {f: sorted(n for n, c in votes[f].items() if c >= maj)
                    for f in votes}
        print(f"{key}: {out[key]}")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
