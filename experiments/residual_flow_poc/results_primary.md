# Residual-flow POC results

Rollouts: 7700; test scenes: 100. Rates are over all flow samples; brackets are 95% Wilson CIs. Paired tests use flow sample 0, paired by scene (exact McNemar).

## Exp 1 - text-conditioned correction (training phrasings, unseen scenes)

| condition | n | viol A (over laptop) | viol B (tilt>15deg) | task success | A-ok & success | B-ok & success | exec. dev. from base (m) | corr. pos (m) | corr. rot (deg) |
|---|---|---|---|---|---|---|---|---|---|
| base | 100 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.00 [0.00,0.04] | 0.00 [0.00,0.04] | 0.000 | 0.000 | 0.0 |
| oracle_A | 100 | 0.03 [0.01,0.08] | 1.00 [0.96,1.00] | 0.98 [0.93,0.99] | 0.97 [0.92,0.99] | 0.00 [0.00,0.04] | 0.064 | 0.059 | 0.0 |
| oracle_B | 100 | 1.00 [0.96,1.00] | 0.00 [0.00,0.04] | 1.00 [0.96,1.00] | 0.00 [0.00,0.04] | 1.00 [0.96,1.00] | 0.003 | 0.000 | 5.6 |
| oracle_AB | 100 | 0.00 [0.00,0.04] | 0.00 [0.00,0.04] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.062 | 0.059 | 5.6 |
| main|A | 300 | 0.04 [0.02,0.06] | 1.00 [0.99,1.00] | 0.99 [0.97,1.00] | 0.96 [0.94,0.98] | 0.00 [0.00,0.01] | 0.063 | 0.059 | 0.0 |
| main|B | 300 | 1.00 [0.99,1.00] | 0.00 [0.00,0.01] | 1.00 [0.99,1.00] | 0.00 [0.00,0.01] | 1.00 [0.99,1.00] | 0.003 | 0.000 | 5.6 |
| main|null | 300 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 0.99 [0.97,1.00] | 0.00 [0.00,0.01] | 0.00 [0.00,0.01] | 0.001 | 0.000 | 0.0 |
| notext | 300 | 0.63 [0.57,0.68] | 0.60 [0.54,0.65] | 1.00 [0.98,1.00] | 0.37 [0.32,0.43] | 0.40 [0.35,0.46] | 0.025 | 0.022 | 2.3 |
| shuffled|A | 300 | 0.63 [0.58,0.69] | 0.59 [0.53,0.64] | 1.00 [0.98,1.00] | 0.37 [0.31,0.42] | 0.41 [0.36,0.47] | 0.025 | 0.022 | 2.3 |
| shuffled|B | 300 | 0.59 [0.53,0.64] | 0.65 [0.59,0.70] | 1.00 [0.98,1.00] | 0.41 [0.36,0.47] | 0.35 [0.30,0.41] | 0.027 | 0.024 | 2.0 |
| mean_A | 100 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.00 [0.00,0.04] | 0.00 [0.00,0.04] | 0.003 | 0.002 | 0.0 |
| mean_B | 100 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.00 [0.00,0.04] | 0.00 [0.00,0.04] | 0.002 | 0.000 | 0.2 |

Paired tests (b = first condition better on that scene, c = second better):

| comparison | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 58 | 0 | 6.9e-18 |
| main|A vs notext | A_ok&succ | 100 | 63 | 2 | 1.2e-16 |
| main|A vs mean_A | A_ok&succ | 100 | 96 | 0 | 2.5e-29 |
| main|B vs shuffled|B | B_ok&succ | 100 | 65 | 0 | 5.4e-20 |
| main|B vs notext | B_ok&succ | 100 | 59 | 0 | 3.5e-18 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 96 | 0 | 2.5e-29 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

Swap specificity (same model, same scenes; only the instruction changes):

| instruction | P(viol A) | P(viol B) | corr. pos (m) | corr. rot (deg) |
|---|---|---|---|---|
| A | 0.04 [0.02,0.06] | 1.00 [0.99,1.00] | 0.059 | 0.0 |
| B | 1.00 [0.99,1.00] | 0.00 [0.00,0.01] | 0.000 | 5.6 |
| null | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 0.000 | 0.0 |

Detour side chosen by main|A agrees with the oracle's side on 93.0% of samples (oracle side is +y on 59% of test scenes).


## Exp 2 - unseen paraphrases

| phrasing | class | max cos to train A | max cos to train B | P(target constraint ok) | P(other constraint still violated) | task success | target-ok & success |
|---|---|---|---|---|---|---|---|
| Don't pass above the electronics. | A | 0.726 | 0.659 | 0.88 [0.80,0.93] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.88 [0.80,0.93] |
| Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 0.85 [0.77,0.91] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.85 [0.77,0.91] |
| Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 0.98 [0.93,0.99] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.98 [0.93,0.99] |
| The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.88 [0.80,0.93] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.88 [0.80,0.93] |
| Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.24 [0.17,0.33] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.24 [0.17,0.33] |
| Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.13 [0.08,0.21] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.13 [0.08,0.21] |
| Go around the laptop. | A | 0.813 | 0.613 | 0.90 [0.83,0.94] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.90 [0.83,0.94] |
| Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.16 [0.10,0.24] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.16 [0.10,0.24] |
| Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |
| Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] |

Pooled A: held-out phrasings 0.63 [0.59,0.66] vs training phrasings 0.96 [0.94,0.98] (A-ok & success).

Pooled B: held-out phrasings 1.00 [1.00,1.00] vs training phrasings 1.00 [0.99,1.00] (B-ok & success).


## Exp 3 - unseen combined instruction A+B

| condition | instruction | viol A | viol B | task success | both ok & success | corr. pos (m) | corr. rot (deg) |
|---|---|---|---|---|---|---|---|
| oracle_AB |  | 0.00 [0.00,0.04] | 0.00 [0.00,0.04] | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.059 | 5.6 |
| main|A | Avoid carrying the mug over the laptop. | 0.04 [0.02,0.06] | 1.00 [0.99,1.00] | 0.99 [0.97,1.00] | 0.00 [0.00,0.01] | 0.059 | 0.0 |
| main|B | Keep the mug upright to prevent spilling. | 1.00 [0.99,1.00] | 0.00 [0.00,0.01] | 1.00 [0.99,1.00] | 0.00 [0.00,0.01] | 0.000 | 5.6 |
| mean_A+mean_B |  | 1.00 [0.96,1.00] | 1.00 [0.96,1.00] | 0.99 [0.95,1.00] | 0.00 [0.00,0.04] | 0.002 | 0.2 |
| main|combined0 | Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.76 [0.70,0.81] | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.040 | 1.1 |
| compose|combined0 | Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.83 [0.78,0.88] | 0.31 [0.25,0.38] | 1.00 [0.98,1.00] | 0.04 [0.02,0.07] | 0.037 | 4.3 |
| shuffled|combined0 | Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.66 [0.59,0.72] | 0.58 [0.52,0.65] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.020 | 2.4 |
| main|combined1 | Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.018 | 2.6 |
| compose|combined1 | Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.84 [0.79,0.89] | 0.44 [0.37,0.51] | 1.00 [0.98,1.00] | 0.01 [0.00,0.04] | 0.038 | 4.1 |
| shuffled|combined1 | Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.62 [0.56,0.69] | 0.58 [0.52,0.65] | 0.99 [0.97,1.00] | 0.00 [0.00,0.02] | 0.022 | 2.3 |
| main|combined2 | Don't pass above the electronics and maintain the cup's vertical orientation. | 1.00 [0.98,1.00] | 0.20 [0.15,0.26] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.005 | 4.3 |
| compose|combined2 | Don't pass above the electronics and maintain the cup's vertical orientation. | 0.99 [0.96,1.00] | 0.05 [0.03,0.09] | 1.00 [0.98,1.00] | 0.01 [0.00,0.04] | 0.024 | 5.2 |
| shuffled|combined2 | Don't pass above the electronics and maintain the cup's vertical orientation. | 0.67 [0.60,0.73] | 0.56 [0.49,0.62] | 0.99 [0.97,1.00] | 0.00 [0.00,0.02] | 0.020 | 2.5 |
| main|combined3 | Carry the mug around the laptop without tilting it. | 1.00 [0.98,1.00] | 0.98 [0.96,0.99] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.015 | 2.9 |
| compose|combined3 | Carry the mug around the laptop without tilting it. | 0.89 [0.84,0.93] | 0.19 [0.14,0.25] | 1.00 [0.98,1.00] | 0.04 [0.02,0.08] | 0.036 | 4.6 |
| shuffled|combined3 | Carry the mug around the laptop without tilting it. | 0.65 [0.58,0.71] | 0.58 [0.52,0.65] | 0.99 [0.96,1.00] | 0.00 [0.00,0.02] | 0.021 | 2.4 |
| main|combined4 | Keep the cup level and away from the space above the laptop. | 0.04 [0.02,0.07] | 1.00 [0.98,1.00] | 0.99 [0.96,1.00] | 0.00 [0.00,0.02] | 0.059 | 0.1 |
| compose|combined4 | Keep the cup level and away from the space above the laptop. | 0.82 [0.77,0.87] | 0.32 [0.25,0.38] | 1.00 [0.98,1.00] | 0.04 [0.02,0.08] | 0.037 | 4.3 |
| shuffled|combined4 | Keep the cup level and away from the space above the laptop. | 0.65 [0.58,0.71] | 0.56 [0.49,0.63] | 0.99 [0.97,1.00] | 0.00 [0.00,0.02] | 0.021 | 2.4 |
| main|combined5 | Do not spill, and do not go over the computer. | 0.53 [0.46,0.60] | 1.00 [0.98,1.00] | 1.00 [0.98,1.00] | 0.00 [0.00,0.02] | 0.049 | 0.7 |
| compose|combined5 | Do not spill, and do not go over the computer. | 0.80 [0.73,0.85] | 0.41 [0.35,0.48] | 1.00 [0.98,1.00] | 0.03 [0.01,0.06] | 0.040 | 4.1 |
| shuffled|combined5 | Do not spill, and do not go over the computer. | 0.62 [0.56,0.69] | 0.65 [0.58,0.71] | 0.99 [0.97,1.00] | 0.00 [0.00,0.02] | 0.023 | 1.9 |

Pooled main|combined*: both-ok&success 0.00 [0.00,0.00], viol A 0.72 [0.69,0.75], viol B 0.86 [0.84,0.88], success 1.00 [0.99,1.00]

Pooled compose|combined*: both-ok&success 0.03 [0.02,0.04], viol A 0.86 [0.84,0.88], viol B 0.29 [0.26,0.31], success 1.00 [1.00,1.00]

Pooled shuffled|combined*: both-ok&success 0.00 [0.00,0.00], viol A 0.65 [0.62,0.67], viol B 0.59 [0.56,0.61], success 0.99 [0.99,1.00]

Paired tests, Exp 3 (AB-ok & success):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main|combined0 vs shuffled|combined0 | 100 | 0 | 0 | 1 |
| compose|combined0 vs main|combined0 | 100 | 3 | 0 | 0.25 |
| main|combined0 vs main|A | 100 | 0 | 0 | 1 |
| main|combined0 vs main|B | 100 | 0 | 0 | 1 |
| main|combined1 vs shuffled|combined1 | 100 | 0 | 0 | 1 |
| compose|combined1 vs main|combined1 | 100 | 2 | 0 | 0.5 |
| main|combined1 vs main|A | 100 | 0 | 0 | 1 |
| main|combined1 vs main|B | 100 | 0 | 0 | 1 |
| main|combined2 vs shuffled|combined2 | 100 | 0 | 0 | 1 |
| compose|combined2 vs main|combined2 | 100 | 1 | 0 | 1 |
| main|combined2 vs main|A | 100 | 0 | 0 | 1 |
| main|combined2 vs main|B | 100 | 0 | 0 | 1 |
| main|combined3 vs shuffled|combined3 | 100 | 0 | 0 | 1 |
| compose|combined3 vs main|combined3 | 100 | 4 | 0 | 0.12 |
| main|combined3 vs main|A | 100 | 0 | 0 | 1 |
| main|combined3 vs main|B | 100 | 0 | 0 | 1 |
| main|combined4 vs shuffled|combined4 | 100 | 0 | 0 | 1 |
| compose|combined4 vs main|combined4 | 100 | 6 | 0 | 0.031 |
| main|combined4 vs main|A | 100 | 0 | 0 | 1 |
| main|combined4 vs main|B | 100 | 0 | 0 | 1 |
| main|combined5 vs shuffled|combined5 | 100 | 0 | 0 | 1 |
| compose|combined5 vs main|combined5 | 100 | 2 | 0 | 0.5 |
| main|combined5 vs main|A | 100 | 0 | 0 | 1 |
| main|combined5 vs main|B | 100 | 0 | 0 | 1 |
