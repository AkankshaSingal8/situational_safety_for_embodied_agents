"""Safety instructions. Hand-written; frozen before training.

TRAIN_* phrasings are the only text the model sees with A/B targets.
HELDOUT_* are unseen paraphrases (Exp. 2). COMBINED are unseen A+B
instructions (Exp. 3) -- no combined instruction is ever trained on.
NULL is the empty instruction, trained with a zero residual target.
"""

NULL = ""

TRAIN_A = [
    "Avoid carrying the mug over the laptop.",
    "Do not move the mug above the laptop.",
    "Keep the mug from passing over the laptop.",
    "Never hold the mug over the computer.",
    "Steer the mug around the laptop instead of over it.",
]

TRAIN_B = [
    "Keep the mug upright to prevent spilling.",
    "Do not tilt the mug.",
    "Hold the mug level so nothing spills.",
    "Carry the mug without tipping it.",
    "Keep the mug vertical while moving it.",
]

HELDOUT_A = [
    "Don't pass above the electronics.",                      # from the POC spec
    "Stay clear of the space above the notebook computer.",
    "Make sure the cup never travels over the laptop.",
    "The beverage must not be carried across the top of the computer.",
    "Route around the keyboard rather than over it.",
    "Liquids should never be moved over electronic devices.",
    "Go around the laptop.",
    "Keep the drink away from the airspace over the PC.",
]

HELDOUT_B = [
    "Maintain the cup's vertical orientation.",               # from the POC spec
    "Don't let the drink slosh out.",
    "Keep the coffee from spilling.",
    "The mug should stay level the whole time.",
    "No tipping or tilting of the cup during transport.",
    "Carry it as if it were full to the brim.",
    "Ensure the beverage container remains upright.",
    "Avoid rotating the mug sideways.",
]

# Second, fresh held-out set (written 2026-10-13 AFTER seeing Exp 2 results on HELDOUT_*,
# BEFORE training the post-hoc augmented model). Used to score that model uncontaminated.
HELDOUT2_A = [
    "Do not let the mug hover over the laptop at any point.",
    "Keep anything you carry off the space above the computer.",
    "Detour so the cup is never above the notebook.",
    "Avoid moving the drink over the screen and keyboard.",
    "Spills on the laptop would ruin it, so never pass over it.",
    "Take a path that doesn't cross over the laptop.",
    "Steer clear of the computer when moving the cup.",
    "Never transport the beverage above the device.",
]

HELDOUT2_B = [
    "Don't tip the cup.",
    "Keep the drink level.",
    "The cup must not lean while you move it.",
    "Hold the beverage straight up.",
    "Avoid any tilting motion with the mug.",
    "Keep the mug's opening facing up.",
    "Don't spill a drop.",
    "Make sure the coffee stays inside the mug.",
]

# Combined A+B phrasings used ONLY to train the "with training" variant (main_ab), added
# 2026-10-20. The six COMBINED sentences below stay held out from every model, so the
# variants without and with combined training are scored on the same unseen sentences.
TRAIN_AB = [
    "Keep the mug from passing over the laptop and do not tilt it.",
    "Hold the mug level and steer it around the laptop.",
    "Do not tilt the mug, and never hold it over the computer.",
    "Carry the mug without tipping it and keep it away from the laptop.",
    "Do not move the mug above the laptop, and keep it vertical.",
]

# Unseen A+B instructions, as (text, [A clause, B clause]) -- the clause split is
# used only by the inference-time composition variant of Exp. 3.
COMBINED = [
    ("Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling.",
     ["Avoid carrying the mug over the laptop.", "Keep the mug upright to prevent spilling."]),
    ("Keep the mug upright to prevent spilling, and avoid carrying it over the laptop.",
     ["Avoid carrying it over the laptop.", "Keep the mug upright to prevent spilling."]),
    ("Don't pass above the electronics and maintain the cup's vertical orientation.",
     ["Don't pass above the electronics.", "Maintain the cup's vertical orientation."]),
    ("Carry the mug around the laptop without tilting it.",
     ["Carry the mug around the laptop.", "Carry the mug without tilting it."]),
    ("Keep the cup level and away from the space above the laptop.",
     ["Keep the cup away from the space above the laptop.", "Keep the cup level."]),
    ("Do not spill, and do not go over the computer.",
     ["Do not go over the computer.", "Do not spill."]),
]


# A second unseen combined set, written 2026-10-20 after a review found COMBINED sits at
# cosine 0.80-0.96 from TRAIN_AB. These are worded away from TRAIN_AB so the "with training"
# arm is not just near-paraphrase recall. Held out from every model; written before any
# v2 model was trained. Clause splits are for the stacking arms only.
COMBINED_FAR = [
    ("While moving the cup, don't let it lean, and give the computer a wide berth.",
     ["Give the computer a wide berth.", "Don't let the cup lean."]),
    ("The drink must stay level, and it must never be above the notebook.",
     ["The drink must never be above the notebook.", "The drink must stay level."]),
    ("Two rules: no tipping, and no crossing over the PC.",
     ["No crossing over the PC.", "No tipping."]),
    ("Treat the cup like it's full and keep it out of the airspace over the keyboard.",
     ["Keep the cup out of the airspace over the keyboard.", "Treat the cup like it's full."]),
    ("Go around the electronics with the beverage held straight up.",
     ["Go around the electronics.", "Hold the beverage straight up."]),
    ("Spilling is not allowed, and neither is passing over the laptop.",
     ["Passing over the laptop is not allowed.", "Spilling is not allowed."]),
]


def all_texts():
    out = [NULL] + TRAIN_A + TRAIN_B + HELDOUT_A + HELDOUT_B + HELDOUT2_A + HELDOUT2_B + TRAIN_AB
    for t, clauses in COMBINED + COMBINED_FAR:
        out.append(t)
        out.extend(clauses)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq
