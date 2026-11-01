# Residual-flow POC v2 results


# Train vs test, headline numbers

| metric | train | test |
|---|---|---|
| base viol A | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| base viol B | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| oracle A+B both ok & succ | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] |
| Exp1 main A text: A ok & succ | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 main A text: B still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 main B text: B ok & succ | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 main B text: A still violated | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
| Exp1 no-text: A ok & succ | 0.335 [0.273,0.403] | 0.410 [0.344,0.479] |
| Exp1 shuffled A text: A ok & succ | 0.370 [0.306,0.439] | 0.425 [0.359,0.494] |
| Exp1 mean A residual: A ok & succ | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] |
| Exp2 heldout A pooled | 0.708 [0.675,0.738] | 0.651 [0.618,0.683] |
| Exp2 heldout B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp2 heldout2 A pooled | 0.632 [0.599,0.665] | 0.615 [0.581,0.648] |
| Exp2 heldout2 B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp3 combined sentence, fed directly (main) | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 velocity composition (main) | 0.092 [0.077,0.109] | 0.092 [0.077,0.110] |
| Exp3 clauses in turn, A then B (main) | 0.987 [0.978,0.992] | 0.984 [0.975,0.990] |
| Exp3 clauses in turn, B then A (main) | 0.984 [0.975,0.990] | 0.989 [0.982,0.994] |
| Exp3 two corrections added (main) | 0.987 [0.978,0.992] | 0.984 [0.975,0.990] |
| Exp3 shuffled-text model | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 trained on combined phrasings (main_ab) | 0.863 [0.842,0.881] | 0.862 [0.841,0.880] |
| Exp3 main_ab, its own training combined phrasings | 1.000 [0.992,1.000] | 1.000 [0.992,1.000] |
| Exp3 far set: sentence fed directly (main) | 0.000 [0.000,0.003] | 0.000 [0.000,0.003] |
| Exp3 far set: clauses in turn, A then B (main) | 0.559 [0.531,0.587] | 0.547 [0.518,0.575] |
| Exp3 far set: clauses in turn, B then A (main) | 0.594 [0.566,0.622] | 0.583 [0.555,0.611] |
| Exp3 far set: two corrections added (main) | 0.555 [0.527,0.583] | 0.546 [0.518,0.574] |
| Exp3 far set: trained on combined phrasings (main_ab) | 0.216 [0.193,0.240] | 0.233 [0.210,0.258] |
| main_ab A text: A ok & succ | 1.000 [0.981,1.000] | 0.995 [0.972,0.999] |
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
| notext | 200 | 0.665 [0.597,0.727] | 0.585 [0.516,0.651] | 1.000 [0.981,1.000] | 0.335 [0.273,0.403] | 0.415 [0.349,0.484] | 0.021 | 2.4 |
| shuffled|A | 200 | 0.630 [0.561,0.694] | 0.585 [0.516,0.651] | 1.000 [0.981,1.000] | 0.370 [0.306,0.439] | 0.415 [0.349,0.484] | 0.023 | 2.2 |
| shuffled|B | 200 | 0.635 [0.566,0.699] | 0.655 [0.587,0.717] | 1.000 [0.981,1.000] | 0.365 [0.301,0.434] | 0.345 [0.283,0.413] | 0.022 | 1.9 |
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
| main|B vs shuffled|B | B_ok&succ | 100 | 67 | 0 | 1.4e-20 |
| main|B vs notext | B_ok&succ | 100 | 61 | 0 | 8.7e-19 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 100.0% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.960 [0.902,0.984] | 1.000 [0.963,1.000] | 0.063 | 0.2 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.067 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.860 [0.779,0.915] | 1.000 [0.963,1.000] | 0.060 | 0.3 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.360 [0.273,0.458] | 1.000 [0.963,1.000] | 0.039 | 0.9 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.260 [0.184,0.354] | 1.000 [0.963,1.000] | 0.035 | 1.1 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.066 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.230 [0.158,0.322] | 1.000 [0.963,1.000] | 0.033 | 1.3 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.061 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.130 [0.078,0.210] | 1.000 [0.963,1.000] | 0.031 | 1.4 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.050 [0.022,0.112] | 1.000 [0.963,1.000] | 0.028 | 1.6 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.061 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.063 | 0.0 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.880 [0.802,0.930] | 1.000 [0.963,1.000] | 0.061 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.470 [0.375,0.567] | 0.008 | 3.7 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.3 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |

- pooled heldout A: 0.708 [0.675,0.738] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.632 [0.599,0.665] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.713 [0.686,0.737] | 0.812 [0.789,0.834] | 0.997 [0.991,0.999] | 0.000 [0.000,0.003] |
| velocity composition (main) | 1200 | 0.671 [0.644,0.697] | 0.335 [0.309,0.362] | 1.000 [0.997,1.000] | 0.092 [0.077,0.109] |
| clauses in turn, A then B (main) | 1200 | 0.010 [0.006,0.017] | 0.003 [0.001,0.009] | 0.998 [0.994,1.000] | 0.987 [0.978,0.992] |
| clauses in turn, B then A (main) | 1200 | 0.012 [0.007,0.019] | 0.004 [0.002,0.010] | 1.000 [0.997,1.000] | 0.984 [0.975,0.990] |
| two corrections added (main) | 1200 | 0.009 [0.005,0.016] | 0.003 [0.001,0.009] | 0.998 [0.993,0.999] | 0.987 [0.978,0.992] |
| shuffled-text model | 1200 | 0.627 [0.600,0.654] | 0.594 [0.566,0.622] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| trained on combined phrasings (main_ab) | 1200 | 0.136 [0.118,0.156] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.863 [0.842,0.881] |
| main_ab, its own training combined phrasings | 500 | 0.000 [0.000,0.008] | 0.000 [0.000,0.008] | 1.000 [0.992,1.000] | 1.000 [0.992,1.000] |
| far set: sentence fed directly (main) | 1200 | 0.833 [0.811,0.853] | 0.369 [0.342,0.397] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| far set: clauses in turn, A then B (main) | 1200 | 0.434 [0.406,0.462] | 0.007 [0.003,0.013] | 0.999 [0.995,1.000] | 0.559 [0.531,0.587] |
| far set: clauses in turn, B then A (main) | 1200 | 0.406 [0.378,0.434] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.594 [0.566,0.622] |
| far set: two corrections added (main) | 1200 | 0.428 [0.401,0.457] | 0.181 [0.160,0.204] | 1.000 [0.997,1.000] | 0.555 [0.527,0.583] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.620 [0.592,0.647] | 0.166 [0.146,0.188] | 1.000 [0.997,1.000] | 0.216 [0.193,0.240] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.00 | 0.08 | 0.99 | 0.98 | 0.98 | 0.99 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.11 | 0.93 | 0.93 | 0.94 | 0.18 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.00 | 0.14 | 0.99 | 0.99 | 0.99 | 1.00 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.08 | 1.00 | 0.99 | 1.00 | 1.00 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.00 | 0.91 | 0.95 | 0.93 | 0.45 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.06 | 0.09 | 0.01 | 0.00 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.47 | 0.52 | 0.47 | 0.82 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.02 | 0.04 | 0.01 | 0.00 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.92 | 0.96 | 0.93 | 0.02 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.98 | 0.99 | 0.98 | 0.01 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 132 | 0 | 3.7e-40 |
| main_ab|far vs stack_AB|far | 600 | 41 | 243 | 4.3e-36 |
| main_ab|combined vs main|combined | 600 | 518 | 0 | 2.3e-156 |
| stack_AB|combined vs main|combined | 600 | 593 | 0 | 6.2e-179 |
| main_ab|combined vs stack_AB|combined | 600 | 2 | 77 | 1e-20 |
| main_ab|combined vs sum|combined | 600 | 4 | 76 | 2.8e-18 |
| main|combined vs shuffled|combined | 600 | 0 | 0 | 1 |

# Split: test (100 scenes, 20500 rollouts)

## Control check and Exp 1 (training phrasings)

| condition | n | viol A | viol B | success | A-ok & succ | B-ok & succ | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| base | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.0 |
| oracle_A | 100 | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.059 | 0.0 |
| oracle_B | 100 | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| oracle_AB | 100 | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.059 | 5.6 |
| main|A | 200 | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.060 | 0.0 |
| main|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.6 |
| main|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |
| notext | 200 | 0.590 [0.521,0.656] | 0.585 [0.516,0.651] | 0.995 [0.972,0.999] | 0.410 [0.344,0.479] | 0.415 [0.349,0.484] | 0.024 | 2.4 |
| shuffled|A | 200 | 0.575 [0.506,0.641] | 0.660 [0.592,0.722] | 1.000 [0.981,1.000] | 0.425 [0.359,0.494] | 0.340 [0.278,0.408] | 0.025 | 2.0 |
| shuffled|B | 200 | 0.625 [0.556,0.689] | 0.575 [0.506,0.641] | 1.000 [0.981,1.000] | 0.375 [0.311,0.444] | 0.425 [0.359,0.494] | 0.022 | 2.4 |
| mean_A | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.001 | 0.0 |
| mean_B | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.3 |
| main_ab|A | 200 | 0.005 [0.001,0.028] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.995 [0.972,0.999] | 0.000 [0.000,0.019] | 0.059 | 0.0 |
| main_ab|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.6 |
| main_ab|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |

| paired comparison (sample 0, by scene) | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 59 | 0 | 3.5e-18 |
| main|A vs notext | A_ok&succ | 100 | 55 | 0 | 5.6e-17 |
| main|A vs mean_A | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs shuffled|B | B_ok&succ | 100 | 61 | 0 | 8.7e-19 |
| main|B vs notext | B_ok&succ | 100 | 64 | 0 | 1.1e-19 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 90.5% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.890 [0.814,0.937] | 1.000 [0.963,1.000] | 0.060 | 0.2 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 0.960 [0.902,0.984] | 1.000 [0.963,1.000] | 0.063 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.790 [0.700,0.858] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.290 [0.210,0.385] | 1.000 [0.963,1.000] | 0.036 | 0.9 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.200 [0.133,0.289] | 1.000 [0.963,1.000] | 0.031 | 1.2 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.960 [0.902,0.984] | 1.000 [0.963,1.000] | 0.063 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.120 [0.070,0.198] | 1.000 [0.963,1.000] | 0.027 | 1.3 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 0.970 [0.915,0.990] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.060 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.110 [0.063,0.186] | 1.000 [0.963,1.000] | 0.027 | 1.4 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.080 [0.041,0.150] | 1.000 [0.963,1.000] | 0.023 | 1.7 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 0.970 [0.915,0.990] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.061 | 0.0 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.810 [0.722,0.875] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.450 [0.356,0.548] | 0.008 | 3.9 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.5 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |

- pooled heldout A: 0.651 [0.618,0.683] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.615 [0.581,0.648] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.735 [0.709,0.759] | 0.825 [0.802,0.845] | 0.998 [0.994,1.000] | 0.000 [0.000,0.003] |
| velocity composition (main) | 1200 | 0.705 [0.679,0.730] | 0.293 [0.268,0.320] | 1.000 [0.997,1.000] | 0.092 [0.077,0.110] |
| clauses in turn, A then B (main) | 1200 | 0.016 [0.010,0.025] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.984 [0.975,0.990] |
| clauses in turn, B then A (main) | 1200 | 0.007 [0.004,0.014] | 0.003 [0.001,0.009] | 1.000 [0.997,1.000] | 0.989 [0.982,0.994] |
| two corrections added (main) | 1200 | 0.016 [0.010,0.025] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.984 [0.975,0.990] |
| shuffled-text model | 1200 | 0.631 [0.603,0.658] | 0.603 [0.575,0.630] | 0.998 [0.994,1.000] | 0.000 [0.000,0.003] |
| trained on combined phrasings (main_ab) | 1200 | 0.138 [0.120,0.159] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.862 [0.841,0.880] |
| main_ab, its own training combined phrasings | 500 | 0.000 [0.000,0.008] | 0.000 [0.000,0.008] | 1.000 [0.992,1.000] | 1.000 [0.992,1.000] |
| far set: sentence fed directly (main) | 1200 | 0.833 [0.811,0.853] | 0.387 [0.360,0.415] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| far set: clauses in turn, A then B (main) | 1200 | 0.453 [0.425,0.482] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.547 [0.518,0.575] |
| far set: clauses in turn, B then A (main) | 1200 | 0.414 [0.387,0.442] | 0.004 [0.002,0.010] | 1.000 [0.997,1.000] | 0.583 [0.555,0.611] |
| far set: two corrections added (main) | 1200 | 0.450 [0.422,0.478] | 0.182 [0.162,0.205] | 1.000 [0.997,1.000] | 0.546 [0.518,0.574] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.610 [0.582,0.637] | 0.163 [0.143,0.185] | 1.000 [0.997,1.000] | 0.233 [0.210,0.258] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.10 | 0.99 | 1.00 | 1.00 | 0.99 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.00 | 0.06 | 1.00 | 1.00 | 1.00 | 0.98 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.14 | 0.92 | 0.94 | 0.91 | 0.21 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.00 | 0.15 | 1.00 | 0.99 | 1.00 | 0.99 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.07 | 0.99 | 0.99 | 0.99 | 0.99 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.00 | 0.92 | 0.96 | 0.92 | 0.47 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.04 | 0.05 | 0.02 | 0.01 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.41 | 0.53 | 0.42 | 0.84 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.92 | 0.96 | 0.93 | 0.05 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.98 | 0.99 | 0.99 | 0.02 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 140 | 0 | 1.4e-42 |
| main_ab|far vs stack_AB|far | 600 | 50 | 240 | 6.2e-31 |
| main_ab|combined vs main|combined | 600 | 518 | 0 | 2.3e-156 |
| stack_AB|combined vs main|combined | 600 | 592 | 0 | 1.2e-178 |
| main_ab|combined vs stack_AB|combined | 600 | 1 | 75 | 2e-21 |
| main_ab|combined vs sum|combined | 600 | 1 | 74 | 4e-21 |
| main|combined vs shuffled|combined | 600 | 0 | 0 | 1 |
