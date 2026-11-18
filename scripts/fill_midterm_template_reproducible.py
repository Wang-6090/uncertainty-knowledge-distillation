"""Generate the midterm report with source-level results and reproduction steps.

The report deliberately reads the saved JSON result files instead of retyping
summary deltas.  This keeps every baseline/treatment value in the document
traceable to a machine-readable artifact.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "附件2-北京交通大学大学生创新创业训练计划项目中期进展报告模板.docx"
OUTPUT = ROOT / "附件2-北京交通大学大学生创新创业训练计划项目中期进展报告_可复现实验修订版.docx"
PROJECT_NAME = "基于不确定性知识蒸馏的新类发现方法"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "未记录"
    quant = Decimal("1").scaleb(-digits)
    rounded = (Decimal(str(value)) * Decimal("100")).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{digits}f}%"


def num(value: float | None, digits: int = 6) -> str:
    return "未记录" if value is None else f"{value:.{digits}f}"


def paired(control: float | None, treatment: float | None, *, percent: bool = False) -> str:
    if control is None or treatment is None:
        return "未记录"
    if percent:
        return f"{pct(control)}→{pct(treatment)}（{100.0 * (treatment - control):+.2f} 个百分点）"
    return f"{num(control)}→{num(treatment)}（{treatment - control:+.6f}）"


def load_shared_results() -> dict[int, dict[str, dict]]:
    result: dict[int, dict[str, dict]] = {}
    for seed in (2081, 2082, 2083):
        result[seed] = {}
        for arm in ("scratch_cnn", "shared_backbone"):
            result[seed][arm] = read_json(
                ROOT / f"runs/rejection_encoder_init_fresh_s{seed}_{arm}_discover/discovery_report.json"
            )
    return result


def load_shared_audits() -> dict[int, dict]:
    result: dict[int, dict] = {}
    for seed in (2081, 2082, 2083):
        result[seed] = read_json(ROOT / f"analysis/rejection_encoder_init_fresh_s{seed}.json")
    return result


def load_uncertainty_results() -> dict[int, dict]:
    return {
        seed: read_json(ROOT / f"analysis/uncertainty_kd_core_s{seed}.json")
        for seed in (2071, 2072, 2073)
    }


def mean(values: list[float | None]) -> float | None:
    present = [x for x in values if x is not None]
    return sum(present) / len(present) if present else None


def shared_result_lines() -> list[str]:
    results = load_shared_results()
    audits = load_shared_audits()
    metrics = [
        ("AUROC", "auroc", False),
        ("AUPR", "aupr", False),
        ("FPR95（越低越好）", "fpr95", True),
        ("OSCR", "oscr", False),
        ("已知类别接受率", "known_accept_rate", True),
        ("未知类别拒识率", "unknown_reject_rate", True),
        ("接受后已知类别准确率", "known_class_accuracy_after_accept", True),
    ]
    lines = [
        "表1  shared-backbone 与 scratch CNN 的逐划分原始结果（shared−scratch；2081—2083）。",
        "每个单元格均按“对照值→shared-backbone值（差值）”记录；百分比指标的差值单位为百分点。",
    ]
    for label, key, is_percent in metrics:
        cells = []
        controls: list[float | None] = []
        treatments: list[float | None] = []
        for seed in (2081, 2082, 2083):
            control = results[seed]["scratch_cnn"].get(key)
            treatment = results[seed]["shared_backbone"].get(key)
            controls.append(control)
            treatments.append(treatment)
            cells.append(f"{seed}: {paired(control, treatment, percent=is_percent)}")
        c_mean = mean(controls)
        t_mean = mean(treatments)
        cells.append(f"均值: {paired(c_mean, t_mean, percent=is_percent)}")
        lines.append(f"{label}：" + "；".join(cells))
    lines.extend(
        [
            "表征审计的原始值：2081 的 known-only rejection train→val probe 为 19.93%→73.04%，有效秩 rank95 为 8→26；",
            "2082 为 17.67%→74.00%，rank95 为 8→27；2083 为 18.41%→73.04%，rank95 为 8→28。",
            "三划分 probe 的均值为 18.67%→73.36%，提高 54.69 个百分点；rank95 均值为 8.00→27.00，提高 19.00。",
            "上述表征数据来自 analysis/rejection_encoder_init_fresh_s2081.json—s2083.json；最终拒识数据来自对应 discovery_report.json。",
            "因果边界：scratch_cnn 与 shared_backbone 同时改变了拒识分支的架构/容量和初始化，因此该实验证明的是 branch configuration 的稳定差异，不是只改变初始化的纯因果效应。",
        ]
    )
    # Keep the audit variable explicitly used so a missing/malformed audit file
    # cannot silently pass while the report is being regenerated.
    for seed in (2081, 2082, 2083):
        if "single_changed_factor" not in audits[seed] and "single_changed_factor_claim" not in audits[seed]:
            raise ValueError(f"missing attribution metadata in seed {seed} audit")
    return lines


def uncertainty_result_lines() -> list[str]:
    results = load_uncertainty_results()
    metrics = [
        ("AUROC", "auroc", False),
        ("AUPR", "aupr", False),
        ("FPR95（越低越好）", "fpr95", False),
        ("OSCR", "oscr", False),
        ("已知类别接受率", "known_accept_rate", True),
        ("未知类别拒识率", "unknown_reject_rate", True),
        ("自动聚类 NMI", "cluster_nmi", False),
        ("自动聚类 ARI", "cluster_ari", False),
    ]
    lines = [
        "表2  训练期 uncertainty-KD 与 standard-KD 的逐划分原始结果（2071—2073）。",
        "control 为 standard-KD，treatment 为 uncertainty-KD；每个单元格均按“standard值→uncertainty值（差值）”记录。",
    ]
    for label, key, is_percent in metrics:
        cells = []
        controls: list[float | None] = []
        treatments: list[float | None] = []
        for seed in (2071, 2072, 2073):
            standard = results[seed]["arms"]["standard"]["metrics"].get(key)
            uncertainty = results[seed]["arms"]["uncertainty"]["metrics"].get(key)
            controls.append(standard)
            treatments.append(uncertainty)
            cells.append(f"{seed}: {paired(standard, uncertainty, percent=is_percent)}")
        cells.append(f"均值: {paired(mean(controls), mean(treatments), percent=is_percent)}")
        lines.append(f"{label}：" + "；".join(cells))
    lines.extend(
        [
            "机制检查：uncertainty-KD 臂每个 epoch 的权重标准差约为 0.1985—0.2258，ESS fraction 约为 0.9513—0.9622；",
            "两臂 logits-KD loss 均非零，feature-KD 系数均为 0，说明该对照实际比较的是 logits-KD 的样本权重方式，而不是未执行的空路径。",
            "结论：2071—2073 中未知类别拒识率分别为 29.63%→22.90%、30.55%→26.80%、28.95%→29.48%；",
            "其中两个划分下降，一个划分仅提高 0.53 个百分点。均值为 29.70%→26.39%，下降 3.32 个百分点；",
            "因此不能将“部分排序指标改善”解释为新类发现整体改善，uncertainty-KD 不作为当前默认训练配置。",
            "原始文件：analysis/uncertainty_kd_core_s2071.json—s2073.json；协议：analysis/uncertainty_kd_core_protocol_20261023.md。",
        ]
    )
    return lines


INTRO = [
    "本项目研究开放环境下的新类发现问题：训练阶段仅对已知类别提供标注，发现或测试阶段可能出现未纳入已知类别集合的新类别。系统需要在没有新类别训练标签的条件下，同时完成已知类别分类、未知样本拒识和新类别聚类。研究重点是检验教师模型不确定性是否能够通过知识蒸馏改善学生模型的新类发现能力。",
    "实验采用 CIFAR-100 semantic-hard 60/40 协议，将 60 个类别划为已知类别，将另外 40 个类别划为新类别；训练阶段使用已知类别标注样本和 5,400 个无标签混合发现池。混合池中的类别身份不作为训练标签，拒识阈值仅由已知验证集校准为目标 95% 已知类别覆盖率。",
    "阶段性结果显示，shared-backbone 拒识分支在三个新划分中具有可复核的原始指标改善；训练期 uncertainty-KD 的权重、损失和梯度均真实生效，但固定覆盖率下未知类别拒识率的原始对比并不支持稳定增益。当前主要困难仍是语义相近的新类别与已知类别在表征和拒识分数上的重叠。",
]


RESEARCH_CONTENT = [
    (
        "（一）主要研究内容",
        [
            "1. 任务与数据协议：已知类别指训练阶段提供类别标注并参与分类监督的类别；新类别指训练阶段未纳入已知类别集合、但在发现或测试阶段出现的类别；混合发现池只提供图像，不提供类别身份。CIFAR-100 semantic-hard 60/40 划分用于构造语义相近、较难拒识的评价条件。",
            "2. 教师—学生分类与标准知识蒸馏：教师模型提供软类别分布，学生模型使用已知类别交叉熵和温度缩放后的 logits KL 蒸馏进行训练。standard-KD 作为对照，用于区分普通知识蒸馏与不确定性加权蒸馏的差异。",
            "3. 不确定性知识蒸馏：对每个样本的 KL 蒸馏损失乘以 exp(-u)，再在 batch 内归一化到平均权重为 1，使实验比较的是样本间蒸馏权重重分配，而非总蒸馏强度变化。分别检查 learned uncertainty 和 MC-epistemic uncertainty 的权重、loss 和梯度是否真实进入训练。",
            "4. 独立拒识表征与 NNPU：拒识分支从学生表征中提取偏离已知支持区域的特征；NNPU 将已知样本作为有标签支持，将混合发现池作为未标记样本参与风险估计，不把混合池直接标记为未知。shared-backbone 配置复制学生 ResNet-18 的表征参数后独立优化，用于解决 scratch CNN 拒识分支的表征容量和优化利用不足。",
            "5. 候选筛选与聚类：使用已知验证集校准拒识阈值，将高风险样本构成候选新类别集合；候选特征使用 PCA 和 KMeans 聚类，auto K 仅依据候选特征内部指标估计，真实类别数量只用于最终回顾性评价。",
            "6. 评价指标：拒识阶段报告 AUROC、AUPR、FPR95、OSCR、已知类别接受率、未知类别拒识率和接受后已知类别准确率；聚类阶段报告自动 K、ACC、NMI 和 ARI。所有方法均以原始 control/treatment 数值、差值和划分条件共同判断，不能只依据平均提升幅度。",
        ],
    )
]


PLAN = [
    "第1阶段（第1—2个月，已完成）：完成开放集新类发现任务定义、相关方法梳理、CIFAR-100 semantic-hard 60/40 划分和基础训练程序。",
    "第2阶段（第3—5个月，已完成）：建立 CE 与 standard-KD 基线，实现 uncertainty-KD，并记录逐 epoch 权重、蒸馏损失和梯度审计结果。",
    "第3阶段（第6—8个月，已完成）：实现独立 rejection encoder、NNPU 风险和 known-validation-only 阈值校准；完成 scratch CNN 与 shared-backbone 的三划分复核。",
    "第4阶段（第9—10个月，已完成主要实验）：完成 validation-only uncertainty fusion、候选筛选和聚类表征对照；使用拒识、已知类别效用和聚类指标联合判定方法是否有效。",
    "中期检查后：固定可复核的 shared-backbone rejection branch 与 standard-KD 对照，优先研究语义相近新类别覆盖和类条件支持表征；训练期 uncertainty-KD 保留为有完整原始对照的消融，不再以单一有利指标推动默认配置。",
]


def reproducibility_lines() -> list[str]:
    return [
        "（三）可复现实验流程",
        "1. 环境：在项目根目录安装 requirements.txt；使用 Python 3.11，数据根目录为 .\\data。实验不下载外部数据，CIFAR-100 数据与本地 split 文件需预先存在。",
        "2. shared-backbone 三划分复核：runner 为 scripts/run_rejection_encoder_init_pair.py。每个 seed 使用同一个 split-matched teacher，并分别训练 scratch_cnn 与 shared_backbone；两臂固定 4 个 student epoch、mixed-pool NNPU、known-validation-only checkpoint、95% known-coverage calibration 和相同 discovery 配置。为避免覆盖已有产物，复现时应使用新的 --tag。",
        "PowerShell 命令（分别执行三个 seed）：",
        "python scripts/run_rejection_encoder_init_pair.py --seed 2081 --split-path splits/semantic_hard/seed2081/cifar100_60_40_semantic_hard.json --source-teacher-config runs/semantic_hard_rejector_seed2028_teacher/config.json --source-student-config runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json --source-discover-config runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json --tag midterm_reproduce_s2081",
        "python scripts/run_rejection_encoder_init_pair.py --seed 2082 --split-path splits/semantic_hard/seed2082/cifar100_60_40_semantic_hard.json --source-teacher-config runs/semantic_hard_rejector_seed2028_teacher/config.json --source-student-config runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json --source-discover-config runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json --tag midterm_reproduce_s2082",
        "python scripts/run_rejection_encoder_init_pair.py --seed 2083 --split-path splits/semantic_hard/seed2083/cifar100_60_40_semantic_hard.json --source-teacher-config runs/semantic_hard_rejector_seed2028_teacher/config.json --source-student-config runs/semantic_hard_vos_boundary_s2040_e4_control/train_student_config.json --source-discover-config runs/semantic_hard_vos_boundary_s2040_e4_control_discover/discover_config.json --tag midterm_reproduce_s2083",
        "3. uncertainty-KD 三划分复核：runner 为 scripts/run_uncertainty_kd_core_pair.py。standard 与 uncertainty 两臂共享 split-matched teacher、学生初始化、训练预算、辅助损失、checkpoint 规则和阈值规则，唯一训练处理因素为 kd_mode；feature KD 和 rejection-branch KD 均关闭。",
        "python scripts/run_uncertainty_kd_core_pair.py --seeds 2071 2072 2073 --source-teacher-config runs/uncertainty_kd_core_s2071_teacher/train_teacher_config.json --source-student-config runs/uncertainty_kd_core_s2071_standard/train_student_config.json --source-discover-config runs/uncertainty_kd_core_s2071_standard_discover/discover_config.json --tag midterm_reproduce_uncertainty_kd",
        "4. 结果核验：shared-backbone 的机器结果读取 runs/rejection_encoder_init_fresh_s2081_scratch_cnn_discover/discovery_report.json 等六个文件；表征审计读取 analysis/rejection_encoder_init_fresh_s2081.json 等三个文件。uncertainty-KD 结果读取 analysis/uncertainty_kd_core_s2071.json—s2073.json。只有在配置审计、loss/权重审计和结果文件均完整时，才将该组实验纳入汇报。",
        "5. 标签隔离：混合池和测试集真实类别只用于最终回顾性统计；不得用于训练标签、checkpoint 选择、阈值校准、自动 K、候选规则、融合权重或方法选择。",
        "6. 当前复现文件的生成：python scripts/fill_midterm_template_reproducible.py。该脚本从上述 JSON 结果重新生成原始值和差值，避免手工转录平均提升数字。",
    ]


PROGRESS = [
    (
        "（一）已开展的工作",
        [
            "1. 完成实验流程模块化，将系统划分为学生分类与蒸馏、拒识表征、候选筛选和新类别聚类四个环节，并为每个环节设置对应 control/treatment 和评价指标。",
            "2. 完成 shared-backbone 三划分原始结果核对。2081 的 AUROC 为 0.776247→0.810519，未知类别拒识率为 20.88%→23.13%；2082 分别为 0.771914→0.798188、24.53%→28.50%；2083 分别为 0.754372→0.789922、19.25%→23.85%。三划分均为同向改善，均值由 AUROC 0.767511 提升至 0.799543（+0.032032，汇总四舍五入为 +0.0320），未知类别拒识率由 21.55% 提升至 25.16%（+3.61 个百分点）。",
            "3. 完成拒识表征审计。2081—2083 的 known-only rejection train→val probe 原始值分别为 19.93%→73.04%、17.67%→74.00%、18.41%→73.04%；有效秩 rank95 分别为 8→26、8→27、8→28。该结果说明 shared-backbone 分支真实恢复了已知类拒识表征，但不能单独证明语义相近新类别的分数重叠已经解决。",
            "4. 完成 uncertainty-KD 三划分原始值核对。2071 的未知类别拒识率为 29.63%→22.90%，2072 为 30.55%→26.80%，2073 为 28.95%→29.48%；均值为 29.70%→26.39%，下降 3.32 个百分点。虽然 AUROC 原始值在 2073 由 0.805646 提升至 0.812663，但 2071、2072 分别下降，因此不能仅依据 AUROC 平均变化认定方法有效。",
            "5. 完成候选筛选、聚类表征和 validation-only uncertainty fusion 对照；这些实验均保留 control/treatment 原始值、差值、划分、校准来源和标签隔离说明，采用拒识、已知类别效用和聚类质量联合判断。",
            "6. 完成程序与实验归档检查。主要实验保存数据划分、配置、checkpoint、训练历史、校准报告、discovery report 和机器可读分析文件；本报告中的结果表由脚本直接读取这些文件生成。",
        ],
    ),
    (
        "（二）当前困难及解决办法",
        [
            "困难一：语义相近的新类别与已知类别在特征空间和拒识分数上重叠。shared-backbone 能改善拒识分支的表征利用，但三划分结果仍不能说明困难新类别已被充分覆盖。后续固定已知类别覆盖率和 known utility，重点检查逐类别候选覆盖与支持区域边界。",
            "困难二：训练期 uncertainty-KD 的机制真实生效，但整体任务结果未形成稳定增益。2071—2073 的逐划分原始 unknown rejection 对比已明确显示两降一升；因此将 uncertainty-KD 保留为消融，不把其作为当前默认训练机制。",
            "困难三：排序指标改善不一定转化为工作点拒识收益。后续每组实验同时报告原始 AUROC、AUPR、FPR95、OSCR、known acceptance、unknown rejection 和 accepted-known accuracy，禁止只报告平均提升幅度。",
            "困难四：候选聚类指标受候选池组成和自动 K 影响。后续把候选污染率、未知覆盖率、逐类覆盖率、自动 K 误差和 ACC/NMI/ARI 一并报告，真实类别数量仅用于最终评价。",
        ],
    ),
    ("（三）可复现实验流程", reproducibility_lines()[1:]),
]


STAGE_RESULTS = [
    "阶段性结果与数据来源：shared-backbone 三个 fresh split 的原始 AUROC 对比为 2081：0.776247→0.810519，2082：0.771914→0.798188，2083：0.754372→0.789922；对应未知类别拒识率为 20.88%→23.13%、24.53%→28.50%、19.25%→23.85%。按 discovery_report.json 重算，三划分均值为 AUROC 0.767511→0.799543（+0.032032），未知类别拒识率 21.55%→25.16%（+3.61 个百分点）。",
    "uncertainty-KD 三个 fresh split 的原始未知类别拒识率为 2071：29.63%→22.90%，2072：30.55%→26.80%，2073：28.95%→29.48%；均值为 29.70%→26.39%（−3.32 个百分点）。因此本阶段支持 shared-backbone 作为拒识表征主线候选，但不支持宣称训练期 uncertainty-KD 已形成稳定新类发现增益。",
    "可复现性边界：所有结果均在固定 semantic-hard 60/40 协议、四 epoch 训练预算、mixed-pool NNPU、known-validation-only checkpoint/95% coverage calibration 下取得；混合池和测试集真实标签未用于训练、模型选择、阈值、自动 K 或方法选择。",
]


def set_font(paragraph, size: float = 10.5, bold: bool = False) -> None:
    for run in paragraph.runs:
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(size)
        run.bold = bold


def clear_cell(cell):
    cell.text = ""
    return cell.paragraphs[0]


def fill_cell(cell, blocks, size: float = 10.0) -> None:
    clear_cell(cell)
    first = True
    for heading, paragraphs in blocks:
        if not first:
            cell.add_paragraph()
        p = cell.paragraphs[0] if first else cell.paragraphs[-1]
        p.text = heading
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_font(p, size=size, bold=True)
        first = False
        for text in paragraphs:
            p = cell.add_paragraph(text)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_font(p, size=size)


def fill_simple_cell(cell, paragraphs, size: float = 10.5) -> None:
    clear_cell(cell)
    for index, text in enumerate(paragraphs):
        p = cell.paragraphs[0] if index == 0 else cell.add_paragraph(text)
        if index == 0:
            p.text = text
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_font(p, size=size)


def main() -> None:
    # Fail early if the report would be generated from incomplete artifacts.
    load_shared_results()
    load_shared_audits()
    load_uncertainty_results()

    doc = Document(TEMPLATE)
    # The template already contains the project information.  Keep the cover,
    # member rows, teacher rows, signatures, and result-form checkboxes intact;
    # this revision only fills the experimental sections requested by the user.

    fill_simple_cell(doc.tables[1].cell(0, 0), INTRO, size=10.3)
    fill_cell(
        doc.tables[2].cell(0, 0),
        RESEARCH_CONTENT
        + [("（二）实施进度安排", PLAN)]
        + [("（三）逐划分原始结果核对", shared_result_lines() + uncertainty_result_lines())],
        size=8.8,
    )
    fill_cell(doc.tables[3].cell(0, 0), PROGRESS, size=8.8)
    fill_simple_cell(doc.tables[4].cell(0, 0), STAGE_RESULTS, size=9.0)

    doc.core_properties.title = PROJECT_NAME + "中期进展报告"
    doc.core_properties.subject = "按模板填写的可复现实验修订版"
    doc.core_properties.comments = (
        "实验数值由机器可读 JSON 和 discovery report 生成；学生、成员和指导教师相关栏目保持模板原状。"
    )
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
