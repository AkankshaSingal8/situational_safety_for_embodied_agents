# Residual flow v2 across training seeds

Runs: runs_v2, runs_v2_s7, runs_v2_s42. Each cell: per-seed values, then mean ± sample s.d. across seeds.


## train scenes

| metric | seed 0 | seed 7 | seed 42 | mean ± s.d. |
|---|---|---|---|---|
| Exp1 A text: laptop avoided & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 A text: tilt still violated (should be 1) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 B text: upright & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 B text: laptop still violated (should be 1) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 no-text model: laptop avoided & placed | 0.340 | 0.335 | 0.335 | 0.337 ± 0.003 |
| Exp1 no-text model: upright & placed | 0.435 | 0.415 | 0.415 | 0.422 ± 0.012 |
| Exp1 shuffled, A text: laptop avoided & placed | 0.365 | 0.370 | 0.360 | 0.365 ± 0.005 |
| Exp1 shuffled, B text: upright & placed | 0.350 | 0.345 | 0.340 | 0.345 ± 0.005 |
| Exp1 detour on the reference side | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 main_ab A text: laptop avoided & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 main_ab B text: upright & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp2 unseen laptop wordings, set 1 | 0.639 | 0.708 | 0.713 | 0.686 ± 0.041 |
| Exp2 unseen laptop wordings, set 2 | 0.634 | 0.632 | 0.646 | 0.637 ± 0.008 |
| Exp2 unseen upright wordings, set 1 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp2 unseen upright wordings, set 2 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp3 close set: sentence given directly | 0.000 | 0.000 | 0.003 | 0.001 ± 0.002 |
| Exp3 close set: velocity composition | 0.097 | 0.092 | 0.114 | 0.101 ± 0.012 |
| Exp3 close set: clauses in turn A then B | 0.976 | 0.987 | 0.983 | 0.982 ± 0.005 |
| Exp3 close set: clauses in turn B then A | 0.975 | 0.984 | 0.985 | 0.981 ± 0.006 |
| Exp3 close set: corrections added | 0.978 | 0.987 | 0.980 | 0.981 ± 0.005 |
| Exp3 close set: trained on combined sentences | 0.874 | 0.863 | 0.873 | 0.870 ± 0.006 |
| Exp3 main_ab on its own training sentences | 0.998 | 1.000 | 0.998 | 0.999 ± 0.001 |
| Exp3 far set: sentence given directly | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| Exp3 far set: clauses in turn A then B | 0.517 | 0.559 | 0.600 | 0.559 ± 0.042 |
| Exp3 far set: clauses in turn B then A | 0.565 | 0.594 | 0.642 | 0.601 ± 0.039 |
| Exp3 far set: corrections added | 0.511 | 0.555 | 0.578 | 0.548 ± 0.034 |
| Exp3 far set: trained on combined sentences | 0.242 | 0.216 | 0.227 | 0.228 ± 0.013 |
| Exp3 far set: corrections added, tilt violated | 0.193 | 0.181 | 0.177 | 0.183 ± 0.008 |
| Exp3 far set: clauses in turn A then B, tilt violated | 0.003 | 0.007 | 0.000 | 0.003 ± 0.003 |

## test scenes

| metric | seed 0 | seed 7 | seed 42 | mean ± s.d. |
|---|---|---|---|---|
| Exp1 A text: laptop avoided & placed | 0.985 | 1.000 | 0.985 | 0.990 ± 0.009 |
| Exp1 A text: tilt still violated (should be 1) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 B text: upright & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 B text: laptop still violated (should be 1) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp1 no-text model: laptop avoided & placed | 0.415 | 0.410 | 0.355 | 0.393 ± 0.033 |
| Exp1 no-text model: upright & placed | 0.415 | 0.415 | 0.410 | 0.413 ± 0.003 |
| Exp1 shuffled, A text: laptop avoided & placed | 0.425 | 0.425 | 0.425 | 0.425 ± 0.000 |
| Exp1 shuffled, B text: upright & placed | 0.455 | 0.425 | 0.430 | 0.437 ± 0.016 |
| Exp1 detour on the reference side | 0.910 | 0.905 | 0.905 | 0.907 ± 0.003 |
| Exp1 main_ab A text: laptop avoided & placed | 0.990 | 0.995 | 1.000 | 0.995 ± 0.005 |
| Exp1 main_ab B text: upright & placed | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp2 unseen laptop wordings, set 1 | 0.651 | 0.651 | 0.718 | 0.673 ± 0.038 |
| Exp2 unseen laptop wordings, set 2 | 0.621 | 0.615 | 0.656 | 0.631 ± 0.022 |
| Exp2 unseen upright wordings, set 1 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp2 unseen upright wordings, set 2 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Exp3 close set: sentence given directly | 0.000 | 0.000 | 0.001 | 0.000 ± 0.000 |
| Exp3 close set: velocity composition | 0.129 | 0.092 | 0.126 | 0.116 ± 0.020 |
| Exp3 close set: clauses in turn A then B | 0.963 | 0.984 | 0.966 | 0.971 ± 0.011 |
| Exp3 close set: clauses in turn B then A | 0.978 | 0.989 | 0.983 | 0.983 ± 0.006 |
| Exp3 close set: corrections added | 0.963 | 0.984 | 0.967 | 0.971 ± 0.011 |
| Exp3 close set: trained on combined sentences | 0.873 | 0.862 | 0.857 | 0.864 ± 0.008 |
| Exp3 main_ab on its own training sentences | 0.992 | 1.000 | 0.996 | 0.996 ± 0.004 |
| Exp3 far set: sentence given directly | 0.000 | 0.000 | 0.001 | 0.000 ± 0.000 |
| Exp3 far set: clauses in turn A then B | 0.529 | 0.547 | 0.598 | 0.558 ± 0.036 |
| Exp3 far set: clauses in turn B then A | 0.587 | 0.583 | 0.660 | 0.610 ± 0.043 |
| Exp3 far set: corrections added | 0.530 | 0.546 | 0.578 | 0.551 ± 0.025 |
| Exp3 far set: trained on combined sentences | 0.253 | 0.233 | 0.237 | 0.241 ± 0.010 |
| Exp3 far set: corrections added, tilt violated | 0.209 | 0.182 | 0.196 | 0.196 ± 0.013 |
| Exp3 far set: clauses in turn A then B, tilt violated | 0.003 | 0.000 | 0.003 | 0.002 ± 0.002 |

## Paired comparisons per seed (test scenes, sample 0; b = first arm better, c = second)

| comparison | seed 0 (b/c, p) | seed 7 (b/c, p) | seed 42 (b/c, p) |
|---|---|---|---|
| main_ab|combined vs main|combined [AB_ok&succ] | 522/0, 1.5e-157 | 518/0, 2.3e-156 | 514/0, 3.7e-155 |
| main_ab|combined vs stack_AB|combined [AB_ok&succ] | 3/58, 3.3e-14 | 1/75, 2e-21 | 7/73, 5.8e-15 |
| main_ab|combined vs sum|combined [AB_ok&succ] | 4/59, 1.4e-13 | 1/74, 4e-21 | 9/73, 1.4e-13 |
| main_ab|far vs main|far [AB_ok&succ] | 154/0, 8.8e-47 | 140/0, 1.4e-42 | 146/0, 2.2e-44 |
| main_ab|far vs stack_AB|far [AB_ok&succ] | 47/212, 3.3e-26 | 50/240, 6.2e-31 | 35/248, 1e-40 |
| main|A vs main|B [A_ok&succ] | 99/0, 3.2e-30 | 100/0, 1.6e-30 | 99/0, 3.2e-30 |
| main|A vs mean_A [A_ok&succ] | 99/0, 3.2e-30 | 100/0, 1.6e-30 | 99/0, 3.2e-30 |
| main|A vs notext [A_ok&succ] | 54/0, 1.1e-16 | 55/0, 5.6e-17 | 59/0, 3.5e-18 |
| main|A vs shuffled|A [A_ok&succ] | 59/1, 1.1e-16 | 59/0, 3.5e-18 | 58/1, 2.1e-16 |
| main|B vs main|A [B_ok&succ] | 100/0, 1.6e-30 | 100/0, 1.6e-30 | 100/0, 1.6e-30 |
| main|B vs mean_B [B_ok&succ] | 100/0, 1.6e-30 | 100/0, 1.6e-30 | 100/0, 1.6e-30 |
| main|B vs notext [B_ok&succ] | 64/0, 1.1e-19 | 64/0, 1.1e-19 | 65/0, 5.4e-20 |
| main|B vs shuffled|B [B_ok&succ] | 55/0, 5.6e-17 | 61/0, 8.7e-19 | 58/0, 6.9e-18 |
| main|combined vs shuffled|combined [AB_ok&succ] | 0/0, 1 | 0/0, 1 | 0/1, 1 |
| stack_AB|combined vs main|combined [AB_ok&succ] | 577/0, 4e-174 | 592/0, 1.2e-178 | 580/0, 5.1e-175 |
