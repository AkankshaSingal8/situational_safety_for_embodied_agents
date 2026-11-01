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
| Exp1 no-text: A ok & succ | 0.335 [0.273,0.403] | 0.355 [0.292,0.423] |
| Exp1 shuffled A text: A ok & succ | 0.360 [0.297,0.429] | 0.425 [0.359,0.494] |
| Exp1 mean A residual: A ok & succ | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] |
| Exp2 heldout A pooled | 0.713 [0.680,0.743] | 0.718 [0.685,0.748] |
| Exp2 heldout B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp2 heldout2 A pooled | 0.646 [0.612,0.679] | 0.656 [0.623,0.688] |
| Exp2 heldout2 B pooled | 1.000 [0.995,1.000] | 1.000 [0.995,1.000] |
| Exp3 combined sentence, fed directly (main) | 0.003 [0.001,0.009] | 0.001 [0.000,0.005] |
| Exp3 velocity composition (main) | 0.114 [0.097,0.133] | 0.126 [0.108,0.146] |
| Exp3 clauses in turn, A then B (main) | 0.983 [0.973,0.989] | 0.966 [0.954,0.975] |
| Exp3 clauses in turn, B then A (main) | 0.985 [0.976,0.990] | 0.983 [0.973,0.989] |
| Exp3 two corrections added (main) | 0.980 [0.970,0.987] | 0.967 [0.955,0.975] |
| Exp3 shuffled-text model | 0.000 [0.000,0.003] | 0.001 [0.000,0.005] |
| Exp3 trained on combined phrasings (main_ab) | 0.873 [0.852,0.890] | 0.857 [0.836,0.875] |
| Exp3 main_ab, its own training combined phrasings | 0.998 [0.989,1.000] | 0.996 [0.986,0.999] |
| Exp3 far set: sentence fed directly (main) | 0.000 [0.000,0.003] | 0.001 [0.000,0.005] |
| Exp3 far set: clauses in turn, A then B (main) | 0.600 [0.572,0.627] | 0.598 [0.570,0.626] |
| Exp3 far set: clauses in turn, B then A (main) | 0.642 [0.615,0.669] | 0.660 [0.633,0.686] |
| Exp3 far set: two corrections added (main) | 0.578 [0.549,0.605] | 0.578 [0.550,0.606] |
| Exp3 far set: trained on combined phrasings (main_ab) | 0.227 [0.204,0.251] | 0.237 [0.214,0.262] |
| main_ab A text: A ok & succ | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] |
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
| notext | 200 | 0.665 [0.597,0.727] | 0.585 [0.516,0.651] | 0.995 [0.972,0.999] | 0.335 [0.273,0.403] | 0.415 [0.349,0.484] | 0.021 | 2.4 |
| shuffled|A | 200 | 0.640 [0.571,0.703] | 0.580 [0.511,0.646] | 1.000 [0.981,1.000] | 0.360 [0.297,0.429] | 0.420 [0.354,0.489] | 0.022 | 2.2 |
| shuffled|B | 200 | 0.640 [0.571,0.703] | 0.660 [0.592,0.722] | 1.000 [0.981,1.000] | 0.360 [0.297,0.429] | 0.340 [0.278,0.408] | 0.022 | 1.9 |
| mean_A | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.001 | 0.0 |
| mean_B | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.3 |
| main_ab|A | 200 | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.062 | 0.0 |
| main_ab|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.4 |
| main_ab|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |

| paired comparison (sample 0, by scene) | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 64 | 0 | 1.1e-19 |
| main|A vs notext | A_ok&succ | 100 | 65 | 0 | 5.4e-20 |
| main|A vs mean_A | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs shuffled|B | B_ok&succ | 100 | 67 | 0 | 1.4e-20 |
| main|B vs notext | B_ok&succ | 100 | 63 | 0 | 2.2e-19 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 100.0% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.920 [0.850,0.959] | 1.000 [0.963,1.000] | 0.061 | 0.2 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 0.980 [0.930,0.994] | 1.000 [0.963,1.000] | 0.064 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.810 [0.722,0.875] | 1.000 [0.963,1.000] | 0.058 | 0.3 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.460 [0.366,0.557] | 1.000 [0.963,1.000] | 0.046 | 0.8 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.330 [0.246,0.427] | 1.000 [0.963,1.000] | 0.041 | 1.1 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.064 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.210 [0.142,0.300] | 1.000 [0.963,1.000] | 0.036 | 1.4 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.3 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.210 [0.142,0.300] | 1.000 [0.963,1.000] | 0.036 | 1.4 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.150 [0.093,0.233] | 1.000 [0.963,1.000] | 0.033 | 1.6 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.062 | 0.0 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.820 [0.733,0.883] | 1.000 [0.963,1.000] | 0.059 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.530 [0.433,0.625] | 0.016 | 3.7 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.4 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.001 | 5.3 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.3 |

- pooled heldout A: 0.713 [0.680,0.743] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.646 [0.612,0.679] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.685 [0.658,0.711] | 0.837 [0.815,0.857] | 0.998 [0.994,1.000] | 0.003 [0.001,0.009] |
| velocity composition (main) | 1200 | 0.608 [0.580,0.636] | 0.372 [0.345,0.399] | 1.000 [0.997,1.000] | 0.114 [0.097,0.133] |
| clauses in turn, A then B (main) | 1200 | 0.018 [0.011,0.027] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.983 [0.973,0.989] |
| clauses in turn, B then A (main) | 1200 | 0.014 [0.009,0.023] | 0.001 [0.000,0.005] | 1.000 [0.997,1.000] | 0.985 [0.976,0.990] |
| two corrections added (main) | 1200 | 0.018 [0.012,0.028] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.980 [0.970,0.987] |
| shuffled-text model | 1200 | 0.627 [0.599,0.654] | 0.598 [0.569,0.625] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| trained on combined phrasings (main_ab) | 1200 | 0.126 [0.108,0.146] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.873 [0.852,0.890] |
| main_ab, its own training combined phrasings | 500 | 0.000 [0.000,0.008] | 0.002 [0.000,0.011] | 1.000 [0.992,1.000] | 0.998 [0.989,1.000] |
| far set: sentence fed directly (main) | 1200 | 0.833 [0.811,0.853] | 0.415 [0.387,0.443] | 1.000 [0.997,1.000] | 0.000 [0.000,0.003] |
| far set: clauses in turn, A then B (main) | 1200 | 0.400 [0.373,0.428] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.600 [0.572,0.627] |
| far set: clauses in turn, B then A (main) | 1200 | 0.357 [0.330,0.384] | 0.001 [0.000,0.005] | 1.000 [0.997,1.000] | 0.642 [0.615,0.669] |
| far set: two corrections added (main) | 1200 | 0.407 [0.380,0.436] | 0.177 [0.156,0.199] | 1.000 [0.997,1.000] | 0.578 [0.549,0.605] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.608 [0.580,0.635] | 0.167 [0.147,0.189] | 1.000 [0.997,1.000] | 0.227 [0.204,0.251] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.10 | 1.00 | 1.00 | 1.00 | 0.99 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.01 | 0.10 | 0.99 | 0.99 | 0.99 | 0.99 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.14 | 0.90 | 0.94 | 0.89 | 0.26 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.01 | 0.15 | 1.00 | 0.97 | 1.00 | 0.99 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.12 | 1.00 | 1.00 | 1.00 | 1.00 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.00 | 0.89 | 0.94 | 0.89 | 0.49 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.15 | 0.20 | 0.10 | 0.02 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.58 | 0.68 | 0.56 | 0.81 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.08 | 0.14 | 0.04 | 0.01 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.91 | 0.90 | 0.90 | 0.02 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.99 | 0.99 | 0.99 | 0.00 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 139 | 0 | 2.9e-42 |
| main_ab|far vs stack_AB|far | 600 | 35 | 258 | 3.7e-43 |
| main_ab|combined vs main|combined | 600 | 525 | 0 | 1.8e-158 |
| stack_AB|combined vs main|combined | 600 | 589 | 0 | 9.9e-178 |
| main_ab|combined vs stack_AB|combined | 600 | 0 | 64 | 1.1e-19 |
| main_ab|combined vs sum|combined | 600 | 1 | 63 | 7e-18 |
| main|combined vs shuffled|combined | 600 | 1 | 0 | 1 |

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
| notext | 200 | 0.645 [0.577,0.708] | 0.590 [0.521,0.656] | 1.000 [0.981,1.000] | 0.355 [0.292,0.423] | 0.410 [0.344,0.479] | 0.021 | 2.4 |
| shuffled|A | 200 | 0.575 [0.506,0.641] | 0.660 [0.592,0.722] | 1.000 [0.981,1.000] | 0.425 [0.359,0.494] | 0.340 [0.278,0.408] | 0.025 | 2.0 |
| shuffled|B | 200 | 0.625 [0.556,0.689] | 0.570 [0.501,0.637] | 1.000 [0.981,1.000] | 0.375 [0.311,0.444] | 0.430 [0.363,0.499] | 0.022 | 2.5 |
| mean_A | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.001 | 0.0 |
| mean_B | 100 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 [0.000,0.037] | 0.000 [0.000,0.037] | 0.000 | 0.3 |
| main_ab|A | 200 | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.060 | 0.0 |
| main_ab|B | 200 | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 1.000 [0.981,1.000] | 0.000 | 5.6 |
| main_ab|null | 200 | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 1.000 [0.981,1.000] | 0.000 [0.000,0.019] | 0.000 [0.000,0.019] | 0.000 | 0.0 |

| paired comparison (sample 0, by scene) | metric | n | b | c | p |
|---|---|---|---|---|---|
| main|A vs shuffled|A | A_ok&succ | 100 | 58 | 1 | 2.1e-16 |
| main|A vs notext | A_ok&succ | 100 | 59 | 0 | 3.5e-18 |
| main|A vs mean_A | A_ok&succ | 100 | 99 | 0 | 3.2e-30 |
| main|B vs shuffled|B | B_ok&succ | 100 | 58 | 0 | 6.9e-18 |
| main|B vs notext | B_ok&succ | 100 | 65 | 0 | 5.4e-20 |
| main|B vs mean_B | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |
| main|A vs main|B | A_ok&succ | 100 | 99 | 0 | 3.2e-30 |
| main|B vs main|A | B_ok&succ | 100 | 100 | 0 | 1.6e-30 |

main|A detours on the oracle's side on 90.5% of samples.

## Exp 2 (unseen paraphrases, model main)

| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |
|---|---|---|---|---|---|---|---|---|
| heldout | Don't pass above the electronics. | A | 0.726 | 0.659 | 0.890 [0.814,0.937] | 1.000 [0.963,1.000] | 0.061 | 0.2 |
| heldout | Stay clear of the space above the notebook computer. | A | 0.755 | 0.590 | 0.980 [0.930,0.994] | 1.000 [0.963,1.000] | 0.062 | 0.1 |
| heldout | Make sure the cup never travels over the laptop. | A | 0.882 | 0.728 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout | The beverage must not be carried across the top of the computer. | A | 0.779 | 0.682 | 0.780 [0.689,0.850] | 1.000 [0.963,1.000] | 0.059 | 0.4 |
| heldout | Route around the keyboard rather than over it. | A | 0.734 | 0.636 | 0.490 [0.394,0.587] | 1.000 [0.963,1.000] | 0.047 | 0.9 |
| heldout | Liquids should never be moved over electronic devices. | A | 0.704 | 0.626 | 0.380 [0.291,0.478] | 1.000 [0.963,1.000] | 0.042 | 1.1 |
| heldout | Go around the laptop. | A | 0.813 | 0.613 | 0.970 [0.915,0.990] | 1.000 [0.963,1.000] | 0.062 | 0.1 |
| heldout | Keep the drink away from the airspace over the PC. | A | 0.821 | 0.750 | 0.260 [0.184,0.354] | 1.000 [0.963,1.000] | 0.038 | 1.4 |
| heldout | Maintain the cup's vertical orientation. | B | 0.659 | 0.862 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Don't let the drink slosh out. | B | 0.712 | 0.773 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Keep the coffee from spilling. | B | 0.788 | 0.868 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | The mug should stay level the whole time. | B | 0.735 | 0.854 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | No tipping or tilting of the cup during transport. | B | 0.645 | 0.791 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Carry it as if it were full to the brim. | B | 0.561 | 0.679 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.001 | 5.5 |
| heldout | Ensure the beverage container remains upright. | B | 0.686 | 0.873 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout | Avoid rotating the mug sideways. | B | 0.804 | 0.882 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Do not let the mug hover over the laptop at any point. | A | 0.944 | 0.827 | 0.980 [0.930,0.994] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout2 | Keep anything you carry off the space above the computer. | A | 0.735 | 0.630 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout2 | Detour so the cup is never above the notebook. | A | 0.759 | 0.696 | 0.260 [0.184,0.354] | 1.000 [0.963,1.000] | 0.036 | 1.5 |
| heldout2 | Avoid moving the drink over the screen and keyboard. | A | 0.797 | 0.739 | 0.210 [0.142,0.300] | 1.000 [0.963,1.000] | 0.034 | 1.6 |
| heldout2 | Spills on the laptop would ruin it, so never pass over it. | A | 0.748 | 0.621 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.059 | 0.0 |
| heldout2 | Take a path that doesn't cross over the laptop. | A | 0.710 | 0.618 | 0.990 [0.946,0.998] | 1.000 [0.963,1.000] | 0.061 | 0.0 |
| heldout2 | Steer clear of the computer when moving the cup. | A | 0.804 | 0.719 | 0.830 [0.745,0.891] | 1.000 [0.963,1.000] | 0.060 | 0.3 |
| heldout2 | Never transport the beverage above the device. | A | 0.740 | 0.688 | 0.000 [0.000,0.037] | 0.490 [0.394,0.587] | 0.015 | 3.8 |
| heldout2 | Don't tip the cup. | B | 0.765 | 0.842 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the drink level. | B | 0.681 | 0.828 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | The cup must not lean while you move it. | B | 0.722 | 0.797 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Hold the beverage straight up. | B | 0.605 | 0.788 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Avoid any tilting motion with the mug. | B | 0.828 | 0.920 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Keep the mug's opening facing up. | B | 0.786 | 0.900 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.6 |
| heldout2 | Don't spill a drop. | B | 0.721 | 0.787 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.001 | 5.5 |
| heldout2 | Make sure the coffee stays inside the mug. | B | 0.761 | 0.844 | 1.000 [0.963,1.000] | 1.000 [0.963,1.000] | 0.000 | 5.5 |

- pooled heldout A: 0.718 [0.685,0.748] (training phrasings: 0.985 [0.957,0.995])
- pooled heldout B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])
- pooled heldout2 A: 0.656 [0.623,0.688] (training phrasings: 0.985 [0.957,0.995])
- pooled heldout2 B: 1.000 [0.995,1.000] (training phrasings: 1.000 [0.981,1.000])

## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)

| arm | n | viol A | viol B | success | both ok & succ |
|---|---|---|---|---|---|
| combined sentence, fed directly (main) | 1200 | 0.662 [0.635,0.689] | 0.838 [0.816,0.858] | 1.000 [0.997,1.000] | 0.001 [0.000,0.005] |
| velocity composition (main) | 1200 | 0.636 [0.608,0.663] | 0.332 [0.306,0.359] | 1.000 [0.997,1.000] | 0.126 [0.108,0.146] |
| clauses in turn, A then B (main) | 1200 | 0.032 [0.023,0.043] | 0.003 [0.001,0.007] | 1.000 [0.997,1.000] | 0.966 [0.954,0.975] |
| clauses in turn, B then A (main) | 1200 | 0.016 [0.010,0.025] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.983 [0.973,0.989] |
| two corrections added (main) | 1200 | 0.031 [0.022,0.042] | 0.003 [0.001,0.009] | 1.000 [0.997,1.000] | 0.967 [0.955,0.975] |
| shuffled-text model | 1200 | 0.632 [0.605,0.659] | 0.608 [0.580,0.635] | 0.998 [0.994,1.000] | 0.001 [0.000,0.005] |
| trained on combined phrasings (main_ab) | 1200 | 0.143 [0.125,0.164] | 0.000 [0.000,0.003] | 1.000 [0.997,1.000] | 0.857 [0.836,0.875] |
| main_ab, its own training combined phrasings | 500 | 0.004 [0.001,0.014] | 0.000 [0.000,0.008] | 1.000 [0.992,1.000] | 0.996 [0.986,0.999] |
| far set: sentence fed directly (main) | 1200 | 0.833 [0.810,0.853] | 0.410 [0.383,0.438] | 1.000 [0.997,1.000] | 0.001 [0.000,0.005] |
| far set: clauses in turn, A then B (main) | 1200 | 0.398 [0.371,0.426] | 0.003 [0.001,0.009] | 1.000 [0.997,1.000] | 0.598 [0.570,0.626] |
| far set: clauses in turn, B then A (main) | 1200 | 0.338 [0.312,0.366] | 0.002 [0.000,0.006] | 1.000 [0.997,1.000] | 0.660 [0.633,0.686] |
| far set: two corrections added (main) | 1200 | 0.402 [0.374,0.430] | 0.196 [0.174,0.219] | 1.000 [0.997,1.000] | 0.578 [0.550,0.606] |
| far set: trained on combined phrasings (main_ab) | 1200 | 0.602 [0.574,0.629] | 0.167 [0.147,0.189] | 1.000 [0.997,1.000] | 0.237 [0.214,0.262] |

Per sentence, both ok & success:

| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|---|
| Avoid carrying the mug over the laptop and keep the mug upright to prevent spilling. | 0.907 | 0.00 | 0.14 | 0.98 | 0.99 | 0.98 | 0.99 |
| Keep the mug upright to prevent spilling, and avoid carrying it over the laptop. | 0.912 | 0.01 | 0.03 | 0.96 | 0.98 | 0.96 | 0.97 |
| Don't pass above the electronics and maintain the cup's vertical orientation. | 0.860 | 0.00 | 0.18 | 0.88 | 0.93 | 0.88 | 0.23 |
| Carry the mug around the laptop without tilting it. | 0.957 | 0.00 | 0.17 | 0.99 | 1.00 | 0.99 | 0.98 |
| Keep the cup level and away from the space above the laptop. | 0.834 | 0.00 | 0.13 | 0.99 | 0.99 | 0.98 | 0.97 |
| Do not spill, and do not go over the computer. | 0.797 | 0.00 | 0.09 | 0.99 | 1.00 | 0.99 | 0.99 |

Far set (worded away from main_ab's training phrasings), both ok & success:

| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |
|---|---|---|---|---|---|---|
| While moving the cup, don't let it lean, and give the computer a wide berth. | 0.733 | 0.01 | 0.88 | 0.92 | 0.88 | 0.49 |
| The drink must stay level, and it must never be above the notebook. | 0.716 | 0.00 | 0.16 | 0.25 | 0.10 | 0.01 |
| Two rules: no tipping, and no crossing over the PC. | 0.703 | 0.00 | 0.60 | 0.69 | 0.59 | 0.86 |
| Treat the cup like it's full and keep it out of the airspace over the keyboard. | 0.729 | 0.00 | 0.10 | 0.20 | 0.06 | 0.01 |
| Go around the electronics with the beverage held straight up. | 0.710 | 0.00 | 0.89 | 0.93 | 0.89 | 0.04 |
| Spilling is not allowed, and neither is passing over the laptop. | 0.728 | 0.00 | 0.96 | 0.97 | 0.96 | 0.00 |

Paired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):

| comparison | n | b | c | p |
|---|---|---|---|---|
| main_ab|far vs main|far | 600 | 146 | 0 | 2.2e-44 |
| main_ab|far vs stack_AB|far | 600 | 35 | 248 | 1e-40 |
| main_ab|combined vs main|combined | 600 | 514 | 0 | 3.7e-155 |
| stack_AB|combined vs main|combined | 600 | 580 | 0 | 5.1e-175 |
| main_ab|combined vs stack_AB|combined | 600 | 7 | 73 | 5.8e-15 |
| main_ab|combined vs sum|combined | 600 | 9 | 73 | 1.4e-13 |
| main|combined vs shuffled|combined | 600 | 0 | 1 | 1 |
