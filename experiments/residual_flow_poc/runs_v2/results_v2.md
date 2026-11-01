# Residual-flow POC v2 results


# Train vs test, headline numbers

| metric | train | test |
|---|---|---|
| base viol A | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| base viol B | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| oracle A+B both ok & succ | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| Exp1 main A text: A ok & succ | 1.000 [0.981,1.000] | 0.985 [0.957,0.995] |
| Exp1 main A text: B still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 main B text: B ok & succ | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 main B text: A still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 no-text: A ok & succ | 0.340 [0.278,0.408] | 0.415 [0.349,0.484] |
| Exp1 shuffled A text: A ok & succ | 0.365 [0.301,0.434] | 0.425 [0.359,0.494] |
| Exp1 mean A residual: A ok & succ | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] |
| Exp2 heldout A pooled | 0.639 [0.605,0.671] | 0.651 [0.618,0.683] |
| Exp2 heldout B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp2 heldout2 A pooled | 0.634 [0.600,0.666] | 0.621 [0.587,0.654] |
| Exp2 heldout2 B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp3 combined sentence, fed directly (main) | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 velocity composition (main) | 0.097 [0.081,0.115] | 0.129 [0.111,0.149] |
| Exp3 clauses in turn, A then B (main) | 0.976 [0.966,0.983] | 0.963 [0.951,0.973] |
| Exp3 clauses in turn, B then A (main) | 0.975 [0.965,0.982] | 0.978 [0.967,0.984] |
| Exp3 two corrections added (main) | 0.978 [0.967,0.984] | 0.963 [0.951,0.973] |
| Exp3 shuffled-text model | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 trained on combined phrasings (main_ab) | 0.874 [0.854,0.892] | 0.873 [0.852,0.890] |
| Exp3 main_ab, its own training combined phrasings | 0.998 [0.989,1.000] | 0.992 [0.980,0.997] |
| Exp3 far set: sentence fed directly (main) | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 far set: clauses in turn, A then B (main) | 0.517 [0.488,0.545] | 0.529 [0.501,0.557] |
| Exp3 far set: clauses in turn, B then A (main) | 0.565 [0.537,0.593] | 0.587 [0.559,0.614] |
| Exp3 far set: two corrections added (main) | 0.511 [0.483,0.539] | 0.530 [0.502,0.558] |
| Exp3 far set: trained on combined phrasings (main_ab) | 0.242 [0.218,0.267] | 0.253 [0.229,0.278] |
| main_ab A text: A ok & succ | 1.000 [0.981,1.000] | 0.990 [0.964,0.997] |
| main_ab A text: B still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| main_ab B text: B ok & succ | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| main_ab B text: A still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |

# Split: train (100 scenes, 20500 rollouts)

## Control check and Exp 1 (training phrasings)

| condition | n | viol A | viol B | success | A-ok & succ | B-ok & succ | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| base | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.0 |
| oracle_A | 100 | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.062 | 0.0 |
| oracle_B | 100 | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| oracle_AB | 100 | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 5.4 |
| main|A | 200 | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.062 | 0.0 |
| main|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.4 |
| main|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |
| notext | 200 | 0.660 [0.592,0.722] | 0.565 [0.496,0.632] | 1.000 [0.981,1.000] | 0.340 [0.278,0.408] | 0.435 [0.368,0.504] | 0.021 | 2.5 |
| shuffled|A | 200 | 0.635 [0.566,0.699] | 0.585 [0.516,0.651] | 1.000 [0.981,1.000] | 0.365 [0.301,0.434] | 0.415 [0.349,0.484] | 0.023 | 2.1 |
| shuffled|B | 200 | 0.635 [0.566,0.699] | 0.650 [0.582,0.713] | 1.000 [0.981,1.000] | 0.365 [0.301,0.434] | 0.350 [0.287,0.418] | 0.022 | 1.9 |
| mean_A | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.001 | 0.0 |
| mean_B | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.3 |
| main_ab|A | 200 | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.062 | 0.0 |
| main_ab|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.4 |
| main_ab|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |

| paired comparison (sample 0, by scene) | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 63 | 0 | 2.2e-19 |
| main|A vs notext | A_ok&succ | 100 | 65 | 0 | 5.4e-20 |
| main|A vs mean_A | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs shuffled|B | B_ok&succ | 100 | 66 | 0 | 2.7e-20 |
| main|B vs notext | B_ok&succ | 100 | 60 | 0 | 1.7e-18 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 100.0% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.880 [0.802,0.930] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.071 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.850 [0.767,0.907] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.190 [0.125,0.278] | 1.000 [0.963,1.000] | 0.036 | 1.1 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.110 [0.063,0.186] | 1.000 [0.963,1.000] | 0.032 | 1.3 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.980 [0.930,0.994] | 1.000 [0.963,1.000] | 0.070 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.100 [0.055,0.174] | 1.000 [0.963,1.000] | 0.029 | 1.4 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.063 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.100 [0.055,0.174] | 1.000 [0.963,1.000] | 0.026 | 1.5 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.100 [0.055,0.174] | 1.000 [0.963,1.000] | 0.023 | 1.6 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.061 | 0.1 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.880 [0.802,0.930] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.390 [0.300,0.488] | 0.005 | 3.8 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.3 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |

- pooled heldout A: 0.639 [0.605,0.671] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.634 [0.600,0.666] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.743 [0.717,0.766] | 0.800 [0.776,0.822] | 0.998 [0.994,1.000] | 0.000 [0.000,0.003] |
| velocity composition (main) | 1200 | 0.634 [0.607,0.661] | 0.354 [0.328,0.382] | 1.000 [0.997,1.000] | 0.097 [0.081,0.115] |
| clauses in turn, A then B (main) | 1200 | 0.022 [0.016,0.033] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.976 [0.966,0.983] |
| clauses in turn, B then A (main) | 1200 | 0.022 [0.015,0.032] | 0.003 [0.001,0.009] | 1.000 [0.997,1.000] | 0.975 [0.965,0.982] |
| two corrections added (main) | 1200 | 0.022 [0.015,0.032] | 0.001 [0.000,0.005] | 1.000 [0.997,1.000] | 0.978 [0.967,0.984] |
| shuffled-text model | 1200 | 0.629 [0.601,0.656] | 0.589 [0.561,0.617] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| trained on combined phrasings (main_ab) | 1200 | 0.125 [0.107,0.145] | 0.001 [0.000,0.005] | 1.000 [0.997,1.000] | 0.874 [0.854,0.892] |
| main_ab, its own training combined phrasings | 500 | 0.000 [0.000,0.008] | 0.002 [0.000,0.011] | 1.000 [0.992,1.000] | 0.998 [0.989,1.000] |
| far set: sentence fed directly (main) | 1200 | 0.833 [0.811,0.853] | 0.367 [0.340,0.394] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| far set: clauses in turn, A then B (main) | 1200 | 0.480 [0.452,0.508] | 0.003 [0.001,0.009] | 0.999 [0.995,1.000] | 0.517 [0.488,0.545] |
| far set: clauses in turn, B then A (main) | 1200 | 0.431 [0.403,0.459] | 0.004 [0.002,0.010] | 1.000 [0.997,1.000] | 0.565 [0.537,0.593] |
| far set: two corrections added (main) | 1200 | 0.482 [0.453,0.510] | 0.193 [0.171,0.216] | 0.998 [0.994,1.000] | 0.511 [0.483,0.539] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.592 [0.564,0.619] | 0.167 [0.147,0.189] | 1.000 [0.997,1.000] | 0.242 [0.218,0.267] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.09 | 1.00 | 0.99 | 1.00 | 1.00 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.00 | 0.14 | 0.99 | 0.99 | 0.99 | 1.00 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.07 | 0.87 | 0.89 | 0.88 | 0.25 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.00 | 0.14 | 0.99 | 0.98 | 1.00 | 1.00 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.07 | 1.00 | 0.99 | 1.00 | 1.00 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.08 | 0.99 | 1.00 | 0.99 | 0.99 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.00 | 0.93 | 0.94 | 0.92 | 0.58 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.05 | 0.13 | 0.04 | 0.01 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.26 | 0.41 | 0.24 | 0.84 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.01 | 0.02 | 0.00 | 0.00 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.88 | 0.91 | 0.88 | 0.02 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.98 | 0.98 | 0.98 | 0.00 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 143 | 0 | 1.8e-43 |
| main_ab|far vs stack_AB|far | 600 | 60 | 231 | 8e-25 |
| main_ab|combined vs main|combined | 600 | 527 | 0 | 4.6e-159 |
| stack_AB|combined vs main|combined | 600 | 587 | 0 | 3.9e-177 |
| main_ab|combined vs stack_AB|combined | 600 | 1 | 61 | 2.7e-17 |
| main_ab|combined vs sum|combined | 600 | 0 | 62 | 4.3e-19 |
| main|combined vs shuffled|combined | 600 | 0 | 0 | 1 |

# Split: test (100 scenes, 20500 rollouts)

## Control check and Exp 1 (training phrasings)

| condition | n | viol A | viol B | success | A-ok & succ | B-ok & succ | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| base | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.0 |
| oracle_A | 100 | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.059 | 0.0 |
| oracle_B | 100 | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| oracle_AB | 100 | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.059 | 5.6 |
| main|A | 200 | 0.015 [0.005,0.043] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.985 [0.957,0.995] | 0.000 [0.000,0.019] | 0.060 | 0.0 |
| main|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.6 |
| main|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |
| notext | 200 | 0.585 [0.516,0.651] | 0.585 [0.516,0.651] | 0.995 [0.972,0.999] | 0.415 [0.349,0.484] | 0.415 [0.349,0.484] | 0.024 | 2.3 |
| shuffled|A | 200 | 0.575 [0.506,0.641] | 0.665 [0.597,0.727] | 1.000 [0.981,1.000] | 0.425 [0.359,0.494] | 0.335 [0.273,0.403] | 0.025 | 1.9 |
| shuffled|B | 200 | 0.635 [0.566,0.699] | 0.545 [0.476,0.613] | 1.000 [0.981,1.000] | 0.365 [0.301,0.434] | 0.455 [0.387,0.524] | 0.022 | 2.6 |
| mean_A | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.001 | 0.0 |
| mean_B | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.3 |
| main_ab|A | 200 | 0.010 [0.003,0.036] | 1.000 [0.981,1.000] | 0.990 [0.964,0.997] | 0.990 [0.964,0.997] | 0.000 [0.000,0.019] | 0.060 | 0.0 |
| main_ab|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.6 |
| main_ab|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |

| paired comparison (sample 0, by scene) | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 59 | 1 | 1.1e-16 |
| main|A vs notext | A_ok&succ | 100 | 54 | 0 | 1.1e-16 |
| main|A vs mean_A | A_ok&succ | 100 | 99 | 0 | 3.2e-30 |
| main|B vs shuffled|B | B_ok&succ | 100 | 55 | 0 | 5.6e-17 |
| main|B vs notext | B_ok&succ | 100 | 64 | 0 | 1.1e-19 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 99 | 0 | 3.2e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 91.0% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.800 [0.711,0.867] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 0.950 [0.888,0.978] | 1.000 [0.963,1.000] | 0.070 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.060 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.770 [0.678,0.842] | 1.000 [0.963,1.000] | 0.057 | 0.3 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.350 [0.264,0.447] | 1.000 [0.963,1.000] | 0.039 | 1.1 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.240 [0.167,0.332] | 1.000 [0.963,1.000] | 0.035 | 1.3 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.930 [0.863,0.966] | 1.000 [0.963,1.000] | 0.067 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.170 [0.109,0.255] | 1.000 [0.963,1.000] | 0.031 | 1.5 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 0.980 [0.930,0.994] | 1.000 [0.963,1.000] | 0.060 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.061 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.120 [0.070,0.198] | 1.000 [0.963,1.000] | 0.029 | 1.6 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.080 [0.041,0.150] | 1.000 [0.963,1.000] | 0.027 | 1.7 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.060 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.061 | 0.1 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.800 [0.711,0.867] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.370 [0.282,0.468] | 0.004 | 4.0 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.001 | 5.5 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |

- pooled heldout A: 0.651 [0.618,0.683] (training phrasings: 0.985 [0.957,0.995])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.621 [0.587,0.654] (training phrasings: 0.985 [0.957,0.995])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.719 [0.693,0.744] | 0.814 [0.791,0.835] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| velocity composition (main) | 1200 | 0.637 [0.610,0.664] | 0.323 [0.297,0.350] | 1.000 [0.997,1.000] | 0.129 [0.111,0.149] |
| clauses in turn, A then B (main) | 1200 | 0.035 [0.026,0.047] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.963 [0.951,0.973] |
| clauses in turn, B then A (main) | 1200 | 0.021 [0.014,0.031] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.978 [0.967,0.984] |
| two corrections added (main) | 1200 | 0.033 [0.025,0.045] | 0.003 [0.001,0.009] | 1.000 [0.997,1.000] | 0.963 [0.951,0.973] |
| shuffled-text model | 1200 | 0.644 [0.617,0.671] | 0.599 [0.571,0.627] | 0.998 [0.993,0.999] | 0.000 [0.000,0.003] |
| trained on combined phrasings (main_ab) | 1200 | 0.128 [0.110,0.148] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.873 [0.852,0.890] |
| main_ab, its own training combined phrasings | 500 | 0.008 [0.003,0.020] | 0.000 [0.000,0.008] | 1.000 [0.992,1.000] | 0.992 [0.980,0.997] |
| far set: sentence fed directly (main) | 1200 | 0.834 [0.812,0.854] | 0.369 [0.342,0.397] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| far set: clauses in turn, A then B (main) | 1200 | 0.469 [0.441,0.497] | 0.003 [0.001,0.007] | 0.999 [0.995,1.000] | 0.529 [0.501,0.557] |
| far set: clauses in turn, B then A (main) | 1200 | 0.411 [0.383,0.439] | 0.004 [0.002,0.010] | 0.999 [0.995,1.000] | 0.587 [0.559,0.614] |
| far set: two corrections added (main) | 1200 | 0.468 [0.439,0.496] | 0.209 [0.187,0.233] | 0.998 [0.994,1.000] | 0.530 [0.502,0.558] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.583 [0.555,0.611] | 0.167 [0.147,0.189] | 1.000 [0.997,1.000] | 0.253 [0.229,0.278] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.13 | 0.99 | 1.00 | 0.98 | 1.00 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.00 | 0.11 | 0.99 | 0.98 | 0.99 | 0.98 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.14 | 0.81 | 0.90 | 0.81 | 0.27 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.00 | 0.16 | 0.99 | 0.99 | 0.99 | 0.98 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.12 | 1.00 | 0.99 | 1.00 | 1.00 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.12 | 0.99 | 1.00 | 0.99 | 1.00 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.00 | 0.89 | 0.93 | 0.91 | 0.57 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.04 | 0.13 | 0.03 | 0.01 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.43 | 0.55 | 0.43 | 0.86 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.01 | 0.04 | 0.01 | 0.01 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.84 | 0.92 | 0.84 | 0.05 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.96 | 0.95 | 0.96 | 0.00 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 154 | 0 | 8.8e-47 |
| main_ab|far vs stack_AB|far | 600 | 47 | 212 | 3.3e-26 |
| main_ab|combined vs main|combined | 600 | 522 | 0 | 1.5e-157 |
| stack_AB|combined vs main|combined | 600 | 577 | 0 | 4e-174 |
| main_ab|combined vs stack_AB|combined | 600 | 3 | 58 | 3.3e-14 |
| main_ab|combined vs sum|combined | 600 | 4 | 59 | 1.4e-13 |
| main|combined vs shuffled|combined | 600 | 0 | 0 | 1 |
