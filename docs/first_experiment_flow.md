# 第一版实验流程（CIFAR-100）

这份流程的目标不是一次做出最强结果，而是先把“数据划分 -> teacher -> uncertainty -> student 蒸馏 -> unknown 检测 -> 聚类发现 -> 评测输出”整条链路跑通。

## 1. 先确定本版实验范围

- 主数据集：CIFAR-100
- 默认 known 类数量：60
- 默认 novel 类数量：40
- 默认随机种子：42
- 默认 backbone：ResNet-18
- 默认输入尺寸：224
- 第一版先只做单次划分、单次训练、单次评测

## 2. 先跑通数据与环境

先确认依赖安装成功：

    pip install -r requirements.txt

再确认 CIFAR-100 能下载并被脚本正确读取：

    python train.py train_teacher --dataset cifar100 --data-root ./data --download --epochs 1 --work-dir ./runs/cifar100_v1

如果这一步能跑完，说明：

- 数据下载正常
- known / novel 划分正常
- 训练脚本能正常启动
- 模型 forward 没有结构错误

## 3. 先做 teacher baseline

第一阶段只训练 teacher，不加 student 蒸馏。

推荐命令：

    python train.py train_teacher --dataset cifar100 --data-root ./data --download --num-known 60 --seed 42 --epochs 20 --batch-size 64 --work-dir ./runs/cifar100_v1

这一阶段要重点看三件事：

- known 类训练是否收敛
- 验证集 known accuracy 是否稳定上升
- checkpoint 和 prototype 是否成功保存

如果结果正常，再把 epochs 提到 50 做正式版。

## 4. 再训练 student

teacher 跑通后，再做不确定性感知蒸馏。

推荐命令：

    python train.py train_student --dataset cifar100 --data-root ./data --download --num-known 60 --seed 42 --epochs 20 --batch-size 64 --work-dir ./runs/cifar100_v1

这一阶段重点看：

- student 是否能继承 teacher 的 known 分类能力
- 蒸馏损失是否正常下降
- 不确定性分支是否能输出合理分数

## 5. 做开放集检测和新类发现

student 训练完后，做未知检测和聚类。

推荐命令：

    python train.py discover --dataset cifar100 --data-root ./data --download --num-known 60 --seed 42 --batch-size 64 --work-dir ./runs/cifar100_v1

这一阶段输出的核心结果是：

- AUROC
- FPR95
- 聚类 ACC
- 聚类 NMI
- 聚类 ARI

## 6. 第一版先要形成的结果文件

跑完后，至少要保留这些内容：

- runs/cifar100_v1/teacher.pt
- runs/cifar100_v1/student.pt
- runs/cifar100_v1/split.pt
- runs/cifar100_v1/discovery_report.json
- runs/cifar100_v1/open_scores.npy

这些文件足够支撑第一轮老师指导。

## 7. 第一版先不做的事

为了避免范围过大，第一版先不追求：

- 多数据集同时跑
- 多种 backbone 对比
- 复杂聚类算法替换
- 复杂阈值自适应策略
- 大规模超参数搜索

这些留到后面老师确认方向之后再扩展。

## 8. 第一版结束后要问老师的三个问题

1. 现在的 known / novel 划分是否合适？
2. 不确定性部分是保留当前形式，还是改成更明确的 Bayesian / ensemble 方案？
3. 新类发现阶段，是否要从 KMeans 升级到更强的聚类策略？

## 9. 最小完成标准

第一版流程只要满足下面三条，就算阶段性成功：

- teacher 和 student 都能正常训练
- unknown 检测能输出 AUROC / FPR95
- unknown 聚类能输出 ACC / NMI / ARI

达到这个程度，你们就可以拿去和老师讨论下一轮改法。
