# Multi-seed Ablation Summary

Results are mean +/- sample standard deviation across completed seeds. Each metric includes its valid report count (n). Older reports may omit newer metrics such as AUPR or OSCR.

| method | K protocol | seeds | AUROC | AUPR | FPR95 | OSCR | known acc | unknown reject | cluster ACC | NMI | ARI | estimated K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_uncertainty_kd | auto | 42, 123, 3407 | 0.5737 ± 0.0340 (n=3) | 0.4127 ± 0.0000 (n=1) | 0.8833 ± 0.0165 (n=3) | 0.2813 ± 0.0000 (n=1) | 0.4143 ± 0.0042 (n=3) | 0.0564 ± 0.0114 (n=3) | 0.1122 ± 0.0998 (n=3) | 0.2423 ± 0.2766 (n=3) | 0.0296 ± 0.0412 (n=3) | 14.3333 ± 21.3620 (n=3) |
| C_uncertainty_kd | oracle | 42, 123, 3407 | 0.5737 ± 0.0340 (n=3) | 0.4127 ± 0.0000 (n=1) | 0.8833 ± 0.0165 (n=3) | 0.2813 ± 0.0000 (n=1) | 0.4143 ± 0.0042 (n=3) | 0.0564 ± 0.0114 (n=3) | 0.2113 ± 0.0145 (n=3) | 0.5293 ± 0.0270 (n=3) | 0.0628 ± 0.0175 (n=3) | 40.0000 ± 0.0000 (n=3) |

Compare B vs C to test uncertainty-aware KD. `oracle` uses the true novel-class count; `auto` estimates K without unknown labels but uses the configured maximum K.
