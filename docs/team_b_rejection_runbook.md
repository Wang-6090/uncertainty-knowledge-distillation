# 李柯颖线执行手册：未知风险估计与候选筛选

本文件是 `docs/parallel_work_division.md` 第 6 节的可执行版本，回答同一个问题：

> 当前未知检测不稳定，主要来自风险分数、阈值、拒识器的表征配置，还是候选池污染？

本线不修改学生蒸馏目标（上游由王笑颜线提供），也不调整聚类算法（下游是齐誉涵线）；
聚类指标只作为"候选池是否还能继续用"的旁证。

## 1. 一键命令

### 1.1 缩小规模 pilot（只有 CPU 时用来自检链路）

```powershell
python scripts/run_team_b_rejection.py --mode pilot `
  --tag pilot_20261123 --seeds 42,123 --shared-teacher `
  --threads 8 --teacher-epochs 2 --student-epochs 2 `
  --limit-train 8000 --image-size 32 --mc-samples 4
```

pilot 会显式缩小训练集、轮数、分辨率和种子数，`--shared-teacher` 还会让两个种子共用同一个教师。
**它的数值不能作为方法证据**，只能证明"训练 → 发现 → 阈值 → 候选池 → 误差分解"这条链是通的。

### 1.2 正式协议（三个公开种子 × 两种拒识器配置）

```powershell
# 先看将要执行的命令
python scripts/run_team_b_rejection.py --mode formal --tag formal_dry --dry-run

# 正式执行
python scripts/run_team_b_rejection.py --mode formal --tag formal_20261124
```

正式默认值：

| 项 | 值 |
| --- | --- |
| 类别划分 | `splits/cifar100_60_40_imagefolder.json`（60 已知 / 40 未知，与 `splits/cifar100_60_40.json` 是同一类别集合，只是用类名表达） |
| 教师 / 学生 | ResNet-34 / ResNet-18 |
| 种子 | 42、123、3407 |
| 风险分数 | `normalized_entropy_mahalanobis` |
| 阈值 | 只用已知验证集，`global` 第 95 百分位 |
| 拒识器配置 | `scratch_cnn` 与 `shared_backbone` 各一遍，其余参数完全相同 |
| 图像尺寸 / 轮数 | 64 / 教师学生各 60 |
| 聚类（旁证） | `feature_pca` + `kmeans`，oracle K |

可覆盖项：`--seeds`、`--image-size`、`--teacher-epochs`、`--student-epochs`、`--limit-train`、
`--mc-samples`、`--threads`、`--device cuda`、`--data-root`、`--split-path`、`--rejection-feature-dim`。

## 2. 产物结构

```text
runs/team_b_rejection/<tag>/                 # 已被 .gitignore 忽略，只留在本机
  manifest.json                              # 本次协议全量快照
  teacher_s<seed>/                           # config.json / command.txt / train.log / teacher.pt
  s<seed>_<mode>_student/                    # config.json / command.txt / train.log / student.pt
  s<seed>_<mode>_discover/
    config.json, discover_config.json, command.txt, train.log
    discovery_report.json                    # AUROC/AUPR/FPR95/OSCR/接受率/候选池/聚类
    discovery_detail.json                    # 逐样本 score / pred_known / pred_class / 标签
    calibration_report.json                  # 阈值、阈值策略、校准已知接受率
    calibration_scores.npz                   # 已知验证分数（不含测试样本、不含标签）
    open_scores.npy
    threshold_stability.json                 # 已知验证分数上的自助阈值波动
    metrics.json                             # 统一记录格式的指标文件
    summary.md                               # 单次实验的六问小结

analysis/team_b_rejection/<tag>/
  summary.md / summary.json                  # 逐 seed 表、均值±标准差、成对对照、方向一致性
  candidate_pool_scratch_cnn.{md,json}
  candidate_pool_shared_backbone.{md,json}
```

两个目录都在 `.gitignore` 里，属于本机实验产物；需要进版本库的是本手册、脚本和划分文件。

## 3. 第 6 节要求与落地方式对照

| 文档要求 | 落地方式 |
| --- | --- |
| 6.2-1/2 风险分数提取、分布、范围、已知/未知重叠 | `candidate_pool_*.md` 第 2 节（mean/median/p90、50 分箱重叠面积、median gap） |
| 6.2-3/4 阈值由哪些已知验证统计量确定、各 seed 阈值 | `calibration_report.json` + `threshold_stability.json` + `summary.md` 逐 seed 表 |
| 6.2-5 已知接受率、未知拒识率、候选数量 | `discovery_report.json` / `metrics.json` |
| 6.2-6 候选池未知召回、纯度、已知污染率 | `candidate_pool_*.md` 第 1 节 |
| 6.2-7 阈值附近样本的跨种子候选标记变化 | `candidate_pool_*.md` 第 3、4 节（q90–q99 邻域、翻转率、翻转类别） |
| 6.3 `scratch_cnn` vs `shared_backbone` 成对对照 | `summary.md` 第 2、4 节（逐 seed 差值 + 方向一致性表） |
| 6.4 有依据时验证 NNPU | 见第 6 节，默认不执行 |
| 6.5 候选池误差分解 | `candidate_pool_*.md` 第 5 节（未知漏检 / 已知污染 / 跨 seed 翻转 / 候选池内聚类） |
| 6.6 交付结果 | 本手册 + `analysis/team_b_rejection/<tag>/CONCLUSION.md` |

## 4. 判读规则

1. 结论必须同时满足：三个正式种子同向，且不是靠牺牲候选纯度或已知污染率换来的；
2. 单种子、单指标改善只能写成"局部信号"；
3. 阈值只能用已知验证集确定，禁止用测试集未知标签挑工作点；
4. 出现"只是继续扫参数""重复已有对照""日志无法证明机制生效"时停止该方向扩展，保留负结果；
5. 默认候选模式（`--candidate-purify none`）下"候选池 = 风险分数高于阈值的全部样本"，
   因此"池召回"和"未知拒识率"是同一个量，不能当作两条独立证据。

## 5. 复现注意

- **数据**：本机 CIFAR-100 以 imagefolder 形式存放在
  `../uncertainty-knowledge-distillation1.0/CIFAR-100-dataset-main`（`train/` 与 `test/` 各 100 类）。
  `--dataset cifar100` 的官方下载在本机不可用，因此统一用 `--dataset imagefolder` +
  同名类集合的划分文件；这一点必须在组内说明，不能和 `--dataset cifar100` 的历史数值混表。
- **`--limit-train`**：对已知训练子集做固定种子的**随机**抽样（`limit_dataset` 用 `randperm`），
  不是取前 N 个样本，因此不会丢掉类别；但它改变了训练集大小，只允许出现在 pilot。
- **教师**：正式协议每个种子训练一个独立教师；`--shared-teacher` 只用于 pilot 省算力。
- **线程**：本机 PyTorch 是 CPU 版，`--threads` 控制 OMP/MKL 线程数（实测 8 线程比 20 线程快）；
  GPU 机器上用 `--device cuda` 并把 `--threads` 设为 0。
- **拒识编码器**：`--independent-rejection-encoder` 必须搭配 `--rejection-feature-dim > 0`，
  运行器默认 64。

## 6. NNPU（6.4）的触发条件与命令

只有先确认两点才验证 NNPU：发现池里同时存在已知和未知样本；把整个未标注池当作未知类会造成
错误监督。验证时固定一组先验比例、损失权重和训练配置，只按三个正式种子各跑一次，不做参数扫描：

```powershell
python train.py discover --dataset imagefolder --data-root <DATA_ROOT> `
  --split-path splits/cifar100_60_40_imagefolder.json --num-known 60 `
  --seed 42 --image-size 64 --batch-size 64 --pretrained `
  --student-ckpt runs/team_b_rejection/<tag>/s42_scratch_cnn_student/student.pt `
  --independent-rejection-encoder --independent-rejection-encoder-init scratch_cnn `
  --rejection-feature-dim 64 `
  --discovery-pool-mode mixed --rejector-training nnpu --rejector-known-prior 0.6 `
  --score-mode feature_rejector_fusion `
  --cluster-k oracle --cluster-feature feature_pca --save-calibration-scores `
  --work-dir runs/team_b_rejection/<tag>/s42_nnpu_discover
```

判据：NNPU 只提高候选召回、却明显降低候选纯度时，不能替换默认流程，记为负结果。

## 7. 当前状态

| 能力 | 状态 |
| --- | --- |
| 训练 / 发现 / 阈值 / 候选池 / 误差分解全链路 | 已跑通（pilot 2 seed × 2 配置；formal 3 seed × 2 配置） |
| 阈值自助稳定性、跨 seed 翻转率、按类别污染 | 已产出，见 `analysis/team_b_rejection/{pilot_20261123,formal_20261124}/` |
| 正式三 seed 结论 | **已完成**，见 `analysis/team_b_rejection/formal_20261124/CONCLUSION.md` |
| NNPU 对照 | 未执行：默认发现池是纯未知池，NNPU 的前提（混合池）不成立 |

### 7.1 正式实验的主要结论（详见 CONCLUSION.md）

- 两种拒识器配置**没有稳定差异**：AUROC `scratch_cnn` 0.5370 ± 0.0061 对
  `shared_backbone` 0.5324 ± 0.0144，三个种子上方向不一致，候选池召回 3/3 变差；
  按停止规则默认保留 `scratch_cnn`。
- 瓶颈在**风险分数排序能力**：已知/未知分数重叠面积 0.89–0.94，AUROC 只有 0.53，
  未知漏检率 95.3%–95.5%。
- **阈值数值稳定但样本级标记不稳定**：阈值 1.978–2.243，自助 std 0.045–0.066，
  但 12.7%–13.0% 的测试样本候选标记会随种子翻转（阈值邻域内 54%，邻域外 8%–9%）。
- **候选池污染是结果不是原因**：池纯度 0.35，其中 64%–66% 的池内样本是已知类被误拒。

### 7.2 pilot（早期自检，只作对照）

pilot 用 2 个种子、2 epoch、8000 张训练图跑通链路，阈值跨 seed 从 1.3874 到 1.7270、
邻域翻转率约 0.41；数值本身不能作为结论。

## 8. 与其他两条线的接口

- 上游（王笑颜线）：拿到固定的学生配置后，按 `manifest.json` 里的 teacher/student 参数复跑即可，
  本线不改蒸馏目标。
- 下游（齐誉涵线）：`s<seed>_<mode>_discover/discovery_detail.json` 里的
  `pred_known` / `score` / `pred_class` 就是冻结后的候选池；聚类阶段不能再用聚类指标回调阈值。

## 9. 正式实验执行记录（2026-11-25）

### 9.1 环境与已完成的产物

| 产物 | 位置 | 说明 |
| --- | --- | --- |
| 三个教师 | `runs/team_b_rejection/formal_20261124/teacher_s{42,123,3407}/teacher.pt` | 各 30 epoch，84.6 MB |
| 六个学生 | 同目录 `s<seed>_<mode>_student/student.pt` | 16 epoch 预算内按 `checkpoint_selection` 选出的最佳 epoch |
| 六次 discover | 同目录 `s<seed>_<mode>_discover/` | `normalized_entropy_mahalanobis` + 已知验证 q95 |
| CUDA 环境 | `.venv_cuda` | torch 2.12.1+cu130 / torchvision 0.27.1+cu130 |
| pilot 全套 | `analysis/team_b_rejection/pilot_20261123/` | 链路自检，不作证据 |

### 9.2 这台机器不能长时间满负荷跑 GPU（重要）

正式实验被打断过多次，根因不是脚本，而是**显卡驱动蓝屏**：

- 系统日志：`BugCheck 0x00000116` = `VIDEO_TDR_FAILURE`；崩溃前 `nvlddmkm` 连刷 50+ 条事件；
- 发生时间：11-24 23:18（当时被误判为"界面卡死"）、11-25 18:42、11-25 21:45；
- 触发条件：**6 个训练进程同时把 RTX 5060 Laptop 压到 85% 以上并持续 1 小时以上**；
- 当前账户**不是管理员**，改不了注册表里的 `TdrDelay`（默认 2 秒超时）。

因此本机经验规则：

1. 同时最多 **3 个**训练进程；单进程跑最稳（实测 2.9 it/s，比 6 路时的 1.5 it/s 还快）；
2. 用**计划任务**而不是普通后台进程启动（见 `scripts/launch_team_b_detached.ps1`），
   避免会话结束时被 Ctrl+C 一起带走；
3. 训练必须开 `--save-epoch-checkpoints`，否则一崩就全丢（`train.py` 只在训练结束写 `student.pt`）；
4. 长跑之前先执行 `powercfg -change -standby-timeout-ac 0` 关掉休眠。

### 9.3 崩溃后的补救：从 epoch 快照还原 student.pt

`scripts/promote_epoch_checkpoints.py` 会读每个 `s*_student/train.log` 里的
`checkpoint_selection`，选出最佳 epoch 的 `student_epoch_<n>.pt` 复制成 `student.pt`，
并写 `promote_report.json` 记录来源（epoch 编号、分数、可用 checkpoint 列表）。

```powershell
python scripts/promote_epoch_checkpoints.py --run-root runs/team_b_rejection/formal_20261124 --dry-run
python scripts/promote_epoch_checkpoints.py --run-root runs/team_b_rejection/formal_20261124
```

**协议偏离（必须在结论里写明）**：正式协议原本写的是学生 30 epoch，实际因为蓝屏只跑到
16 epoch 就重启用快照还原。六个配置**统一使用 16 epoch 预算**，并在 1–16 内按
`checkpoint_selection` 选最佳 epoch，所以配置之间仍然可比；但不能和 30 epoch 的结果混表。

### 9.4 恢复/重跑命令

```powershell
# 六个配置（3 个种子 × 2 种拒识器）分离式训练，最多 3 个并发更稳
.\scripts\launch_team_b_detached.ps1 -Tag formal_20261124 -Launch

# 查看
Get-ScheduledTask -TaskName 'dsh_tb_formal_20261124_*' | Select TaskName,State
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

# 全部训练结束后：聚合 + 写报告
python scripts/run_team_b_rejection.py --mode formal --tag formal_20261124 `
  --seeds 42,123,3407 --student-epochs 16 `
  --device cuda --threads 2 --num-workers 0 --image-size 64 --mc-samples 8 --skip-existing
```

聚合会在 `analysis/team_b_rejection/formal_20261124/` 下生成 `summary.md`、`summary.json`、
`candidate_pool_scratch_cnn.md`、`candidate_pool_shared_backbone.md`；结论写在同目录
`CONCLUSION.md`。