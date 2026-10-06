# Vietlott walk-forward strategy benchmark

All methods are scored on the same chronological target draws, beginning at position 201. Each strategy sees only draws before its target. Random baseline expectations use the exact Hypergeometric(N, k, k) distribution.

Exact p-values are two-sided tests of cumulative hits against the convolved per-draw hypergeometric null; Holm-Bonferroni adjustment is applied across all method/game comparisons in this report.

## Lotto 5/35 (5_35)

Draws: 796; evaluation targets: 201–796 (596).

| Method | Avg hits/draw | Total hits | Difference vs random expected | Hit distribution (0/5–5/5) | Exact p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| Random combination | 0.6242 | 372 | -53.71 | 0: 298, 1: 232, 2: 59, 3: 6, 4: 1, 5: 0 | 0.0029 | 0.0343 |
| Historical frequency | 0.7081 | 422 | -3.71 | 0: 265, 1: 247, 2: 77, 3: 7, 4: 0, 5: 0 | 0.8454 | 1.0000 |
| Longest-gap / recency | 0.7299 | 435 | +9.29 | 0: 248, 1: 268, 2: 73, 3: 7, 4: 0, 5: 0 | 0.6159 | 1.0000 |
| Current Random Forest | 0.7500 | 447 | +21.29 | 0: 250, 1: 250, 2: 91, 3: 5, 4: 0, 5: 0 | 0.2418 | 1.0000 |

Random baseline mean: 0.7143 hits/draw (425.71 over this evaluation). Expected hit distribution (not observed): 0: 261.63, 1: 251.57, 2: 74.54, 3: 7.99, 4: 0.28, 5: 0.00.

No method is designated predictive unless its Holm-adjusted p-value is below 0.05; even then, independent confirmation would be needed.

## Mega 6/45 (6_45)

Draws: 1369; evaluation targets: 201–1369 (1169).

| Method | Avg hits/draw | Total hits | Difference vs random expected | Hit distribution (0/6–6/6) | Exact p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| Random combination | 0.7938 | 928 | -7.20 | 0: 469, 1: 499, 2: 176, 3: 23, 4: 2, 5: 0, 6: 0 | 0.7940 | 1.0000 |
| Historical frequency | 0.7947 | 929 | -6.20 | 0: 477, 1: 488, 2: 171, 3: 33, 4: 0, 5: 0, 6: 0 | 0.8229 | 1.0000 |
| Longest-gap / recency | 0.8015 | 937 | +1.80 | 0: 464, 1: 495, 2: 190, 3: 18, 4: 2, 5: 0, 6: 0 | 0.9554 | 1.0000 |
| Current Random Forest | 0.7827 | 915 | -20.20 | 0: 495, 1: 469, 2: 170, 3: 34, 4: 1, 5: 0, 6: 0 | 0.4556 | 1.0000 |

Random baseline mean: 0.8000 hits/draw (935.20 over this evaluation). Expected hit distribution (not observed): 0: 468.26, 1: 495.80, 2: 177.07, 3: 26.23, 4: 1.60, 5: 0.03, 6: 0.00.

No method is designated predictive unless its Holm-adjusted p-value is below 0.05; even then, independent confirmation would be needed.

## Power 6/55 (6_55)

Draws: 1401; evaluation targets: 201–1401 (1201).

| Method | Avg hits/draw | Total hits | Difference vs random expected | Hit distribution (0/6–6/6) | Exact p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| Random combination | 0.6411 | 770 | -16.11 | 0: 596, 1: 459, 2: 129, 3: 15, 4: 2, 5: 0, 6: 0 | 0.5258 | 1.0000 |
| Historical frequency | 0.6378 | 766 | -20.11 | 0: 586, 1: 477, 2: 125, 3: 13, 4: 0, 5: 0, 6: 0 | 0.4277 | 1.0000 |
| Longest-gap / recency | 0.6403 | 769 | -17.11 | 0: 593, 1: 462, 2: 132, 3: 13, 4: 1, 5: 0, 6: 0 | 0.5002 | 1.0000 |
| Current Random Forest | 0.6719 | 807 | +20.89 | 0: 568, 1: 476, 2: 140, 3: 17, 4: 0, 5: 0, 6: 0 | 0.4161 | 1.0000 |

Random baseline mean: 0.6545 hits/draw (786.11 over this evaluation). Expected hit distribution (not observed): 0: 579.33, 1: 474.00, 2: 131.67, 3: 15.27, 4: 0.73, 5: 0.01, 6: 0.00.

No method is designated predictive unless its Holm-adjusted p-value is below 0.05; even then, independent confirmation would be needed.
