# 李柯颖线：本机实验结果索引

执行手册（命令、产物结构、第 6 节逐项对照、判读规则、NNPU 触发条件）在
[`docs/team_b_rejection_runbook.md`](../../docs/team_b_rejection_runbook.md)。

本目录只存放本机运行产物（`analysis/` 与 `runs/` 都在 `.gitignore` 中，不会进版本库）：

| 目录 | 内容 |
| --- | --- |
| `pilot_20261123/` | pilot 自检结果：`summary.md`（逐 seed + 成对对照 + 方向一致性）、`candidate_pool_scratch_cnn.md`、`candidate_pool_shared_backbone.md`、`CONCLUSION.md` |

重新生成（不重新训练）：

```powershell
python scripts/run_team_b_rejection.py --mode pilot --tag pilot_20261123 `
  --seeds 42,123 --shared-teacher --skip-existing `
  --threads 8 --teacher-epochs 2 --student-epochs 2 `
  --limit-train 8000 --image-size 32 --mc-samples 4
```

`--skip-existing` 会复用已存在的 teacher / student / discover 产物，只重算分析和汇总。
