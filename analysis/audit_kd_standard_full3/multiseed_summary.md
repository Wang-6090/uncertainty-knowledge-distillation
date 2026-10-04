# Multi-seed Ablation Summary

Results are mean +/- sample standard deviation across completed seeds. Each metric includes its valid report count (n). Older reports may omit newer metrics such as AUPR or OSCR.

| method | K protocol | seeds | AUROC | AUPR | FPR95 | OSCR | known acc | unknown reject | cluster ACC | NMI | ARI | estimated K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_standard_kd | auto | 42, 123, 3407 | 0.5764 ± 0.0233 (n=3) | 0.4252 ± 0.0000 (n=1) | 0.8826 ± 0.0174 (n=3) | 0.2962 ± 0.0000 (n=1) | 0.4261 ± 0.0055 (n=3) | 0.0561 ± 0.0064 (n=3) | 0.1334 ± 0.1198 (n=3) | 0.2577 ± 0.2607 (n=3) | 0.0375 ± 0.0471 (n=3) | 16.0000 ± 24.2487 (n=3) |
| B_standard_kd | oracle | 42, 123, 3407 | 0.5764 ± 0.0233 (n=3) | 0.4252 ± 0.0000 (n=1) | 0.8826 ± 0.0174 (n=3) | 0.2962 ± 0.0000 (n=1) | 0.4261 ± 0.0055 (n=3) | 0.0561 ± 0.0064 (n=3) | 0.2213 ± 0.0423 (n=3) | 0.5399 ± 0.0176 (n=3) | 0.0702 ± 0.0238 (n=3) | 40.0000 ± 0.0000 (n=3) |

Compare B vs C to test uncertainty-aware KD. `oracle` uses the true novel-class count; `auto` estimates K without unknown labels but uses the configured maximum K.
