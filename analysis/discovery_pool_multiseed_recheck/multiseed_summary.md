# Multi-seed Ablation Summary

Results are mean +/- sample standard deviation across completed seeds.

| method | K protocol | seeds | AUROC | AUPR | FPR95 | OSCR | known acc | unknown reject | cluster ACC | NMI | ARI | estimated K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F_discovery_pool | auto | 42, 123 | 0.6397 ± 0.0031 | 0.5021 ± 0.0063 | 0.8413 ± 0.0016 | 0.4084 ± 0.0069 | 0.5221 ± 0.0107 | 0.0864 ± 0.0062 | 0.1258 ± 0.0008 | 0.3389 ± 0.0103 | 0.0426 ± 0.0067 | 11.5000 ± 0.7071 |
| F_discovery_pool | oracle | 42, 123 | 0.6397 ± 0.0031 | 0.5021 ± 0.0063 | 0.8413 ± 0.0016 | 0.4084 ± 0.0069 | 0.5221 ± 0.0107 | 0.0864 ± 0.0062 | 0.2058 ± 0.0070 | 0.5053 ± 0.0101 | 0.0679 ± 0.0093 | 40.0000 ± 0.0000 |

Compare B vs C to test uncertainty-aware KD. `oracle` uses the true novel-class count; `auto` estimates K without unknown labels but uses the configured maximum K.
