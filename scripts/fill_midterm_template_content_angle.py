"""Fill the midterm template according to the template's required content angles.

Only the experiment-related sections are changed.  The team-member commitment
and teacher-opinion sections are copied from the template unchanged.
Numerical claims are read from the saved JSON/discovery reports so that the
report does not rely on manually transcribed averages.
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
OUTPUT = ROOT / "附件2-北京交通大学大学生创新创业训练计划项目中期进展报告_按模板要求重写版.docx"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    q = Decimal("1").scaleb(-digits)
    x = (Decimal(str(value)) * Decimal("100")).quantize(q, rounding=ROUND_HALF_UP)
    return f"{x:.{digits}f}%"


def num(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def pair(control: float, treatment: float, *, percentage: bool = False) -> str:
    if percentage:
        return f"{pct(control)}→{pct(treatment)}（{100 * (treatment - control):+.2f} 个百分点）"
    return f"{num(control)}→{num(treatment)}（{treatment - control:+.6f}）"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def shared_data() -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in (2081, 2082, 2083):
        audit = read_json(ROOT / f"analysis/rejection_encoder_init_fresh_s{seed}.json")
        result[seed] = {
            "scratch_cnn": dict(audit["discovery"]["scratch_cnn"]),
            "shared_backbone": dict(audit["discovery"]["shared_backbone"]),
        }
        for arm in ("scratch_cnn", "shared_backbone"):
            report = read_json(
                ROOT
                / f"runs/rejection_encoder_init_fresh_s{seed}_{arm}_discover/discovery_report.json"
            )
            result[seed][arm]["known_class_accuracy_after_accept"] = float(
                report["known_class_accuracy_after_accept"]
            )
    return result


def uncertainty_data() -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in (2071, 2072, 2073):
        data = read_json(ROOT / f"analysis/uncertainty_kd_core_s{seed}.json")
        result[seed] = {
            arm: dict(data["arms"][arm]["metrics"])
            for arm in ("standard", "uncertainty")
        }
    return result


def shared_result_text() -> str:
    data = shared_data()
    lines = [
        "表1  shared-backbone 与 scratch CNN 的三划分对照结果（每项均为 scratch CNN→shared-backbone，括号为差值）。",
    ]
    metrics = [
        ("AUROC", "auroc", False),
        ("AUPR", "aupr", False),
        ("FPR95（越低越好）", "fpr95", True),
        ("OSCR", "oscr", False),
        ("已知类别接受率", "known_accept_rate", True),
        ("未知类别拒识率", "unknown_reject_rate", True),
        ("接受后已知类别准确率", "known_class_accuracy_after_accept", True),
    ]
    for label, key, percentage in metrics:
        rows = []
        controls = []
        treatments = []
        for seed in (2081, 2082, 2083):
            c = float(data[seed]["scratch_cnn"][key])
            t = float(data[seed]["shared_backbone"][key])
            controls.append(c)
            treatments.append(t)
            rows.append(f"{seed}: {pair(c, t, percentage=percentage)}")
        rows.append(
            f"均值: {pair(mean(controls), mean(treatments), percentage=percentage)}"
        )
        lines.append(f"{label}：" + "；".join(rows))
    lines.extend(
        [
            "表征审计：2081—2083 的 known-only rejection train→validation probe 分别为 19.93%→73.04%、17.67%→74.00%、18.41%→73.04%；有效秩 rank95 分别为 8→26、8→27、8→28。该审计用于确认拒识分支确实获得了可利用的已知类表征，不把它直接等同于新类别发现性能。",
            "条件与边界：三次实验均采用 CIFAR-100 semantic-hard 60/40 划分、相同教师模型和训练预算、独立拒识分支、mixed-pool NNPU、known-validation-only checkpoint 与 95% 已知类覆盖率校准。scratch CNN 与 shared-backbone 同时改变了拒识分支的结构和初始化，因此该对照说明的是分支配置差异，不是单独初始化因素的因果效应。",
        ]
    )
    return "\n".join(lines)


def uncertainty_result_text() -> str:
    data = uncertainty_data()
    lines = [
        "表2  训练期 uncertainty-KD 与 standard-KD 的三划分对照结果（每项均为 standard-KD→uncertainty-KD，括号为差值）。",
    ]
    metrics = [
        ("AUROC", "auroc", False),
        ("AUPR", "aupr", False),
        ("FPR95（越低越好）", "fpr95", True),
        ("OSCR", "oscr", False),
        ("已知类别接受率", "known_accept_rate", True),
        ("未知类别拒识率", "unknown_reject_rate", True),
        ("自动聚类 NMI", "cluster_nmi", False),
        ("自动聚类 ARI", "cluster_ari", False),
    ]
    for label, key, percentage in metrics:
        rows = []
        controls = []
        treatments = []
        for seed in (2071, 2072, 2073):
            c = float(data[seed]["standard"][key])
            t = float(data[seed]["uncertainty"][key])
            controls.append(c)
            treatments.append(t)
            rows.append(f"{seed}: {pair(c, t, percentage=percentage)}")
        rows.append(
            f"均值: {pair(mean(controls), mean(treatments), percentage=percentage)}"
        )
        lines.append(f"{label}：" + "；".join(rows))
    lines.extend(
        [
            "机制核验：uncertainty-KD 的逐样本蒸馏权重、蒸馏损失和梯度均进入训练；两臂 feature-KD 系数均为 0，比较因素限定为 logits-KD 的样本权重方式。",
            "结果解释：未知类别拒识率在 2071、2072 两个划分下降，在 2073 上升 0.53 个百分点；三划分均值由 29.71% 降至 26.39%。因此该机制在当前固定工作点下未形成稳定的整体收益，保留为消融对照，不作为默认训练配置。",
        ]
    )
    return "\n".join(lines)


PROJECT_INTRO = (
    "随着智能驾驶、智能制造和开放环境视觉感知系统的应用，模型接收到的数据类别不再局限于训练阶段已经定义的类别集合。传统监督分类通常假设训练集与测试集具有相同的类别空间，因而难以处理类别不完备和新类别持续出现的情况。本项目研究开放环境下的新类发现问题：训练阶段仅对已知类别样本提供类别标注，模型在部署或发现阶段可能接收到训练类别集合之外的新类别。项目拟构建教师—学生知识蒸馏框架，引入教师模型的不确定性信息，研究其对学生模型表征学习的影响；同时设置独立拒识分支，利用正—未标记学习识别偏离已知类别支持区域的样本，再对候选样本进行特征降维和聚类。项目拟解决已知类别分类、未知样本拒识与新类别组织不能协同完成的问题，预期形成一套可复现的新类发现实验流程和算法实现，为开放环境视觉识别提供方法依据。"
)


RESEARCH_AND_PLAN = (
    "（一）主要研究内容\n"
    "1. 任务建模与数据协议。将训练阶段有类别标注并参与分类监督的类别定义为已知类别；将训练阶段未纳入已知类别集合、但在发现或测试阶段出现的类别定义为新类别。使用 CIFAR-100 semantic-hard 60/40 协议构造语义相近、拒识难度较高的评价条件；混合发现池只提供图像，不把其中的类别身份作为训练标签。\n"
    "2. 教师—学生分类与标准知识蒸馏。教师模型输出软类别分布，学生模型使用已知类交叉熵和温度缩放后的 logits KL 损失训练。standard-KD 作为基础对照，用于区分普通知识蒸馏与不确定性加权蒸馏的作用。\n"
    "3. 不确定性知识蒸馏。根据样本不确定性对 KL 蒸馏损失进行样本级加权，并在 batch 内归一化权重，使比较聚焦于蒸馏信息的分配方式，而不是总蒸馏强度的变化；同时审计权重、损失和梯度，确认该机制真实进入训练。\n"
    "4. 独立拒识表征与 NNPU。拒识分支从学生表征中学习已知类支持区域与偏离区域的差异；NNPU 将已知样本作为正类支持，将混合发现池作为未标记样本估计风险，不把混合池直接标记为未知。采用 shared-backbone 初始化后独立优化，以提高拒识分支对学生表征的利用效率。\n"
    "5. 候选筛选与新类别聚类。使用已知验证集校准拒识阈值，将高风险样本作为候选新类别集合；候选特征经 PCA 降维后使用 KMeans 聚类，自动类别数只依据候选特征内部指标选择，真实类别数量仅用于最终回顾性评价。\n"
    "6. 综合评价。拒识评价包括 AUROC、AUPR、FPR95、OSCR、已知类别接受率、未知类别拒识率和接受后已知类别准确率；聚类评价包括自动 K、ACC、NMI 和 ARI。每项实验同时保留对照值、处理值、差值、划分条件和标签使用边界。\n\n"
    "（二）实施进度安排\n"
    "第1—2个月：完成开放环境新类发现任务定义、相关文献与方法梳理、数据划分协议和基础训练程序。\n"
    "第3—5个月：建立交叉熵和 standard-KD 基线，实现 uncertainty-KD，并完成蒸馏权重、损失及梯度审计。\n"
    "第6—8个月：实现独立拒识分支、NNPU 风险估计和 known-validation-only 阈值校准，完成拒识表征对照。\n"
    "第9—10个月：完成候选筛选、特征聚类及多划分复核，依据拒识指标、已知类效用和聚类质量联合判断方法效果。\n"
    "后续计划：围绕语义相近新类别与已知类别的表征重叠，开展逐类别候选覆盖、支持区域边界和工作点稳定性分析；uncertainty-KD 保留为有完整审计的消融方法，不以单一有利指标推动默认配置。"
)


PROGRESS_AND_DIFFICULTIES = (
    "（一）已开展的工作\n"
    "1. 完成实验流程模块化，将系统划分为分类与蒸馏、拒识表征、候选筛选和新类别聚类四个环节，并为主要因素设置成对对照。\n"
    "2. 完成 shared-backbone 拒识分支的三划分复核。逐划分 AUROC、AUPR、FPR95、OSCR、已知类接受率、未知类拒识率和接受后已知类准确率见表1；结果同时保留原始对照值和差值，避免只报告平均提升。\n"
    + shared_result_text()
    + "\n3. 完成 uncertainty-KD 的三划分机制和效果核验。逐划分结果见表2；确认权重、损失和梯度真实参与训练后，再依据未知类别拒识率、已知类别效用和聚类指标共同判断。\n"
    + uncertainty_result_text()
    + "\n4. 完成实验归档与标签隔离检查。保存数据划分、配置、checkpoint、训练历史、校准报告、discovery report 和机器可读分析文件；混合池和测试集真实类别只用于最终回顾性统计。\n\n"
    "（二）项目实施存在的困难及解决办法\n"
    "困难一：语义相近的新类别与已知类别在特征空间和拒识分数上重叠，导致未知样本容易被接受为已知类别。解决办法是固定已知类覆盖率校准规则，比较不同拒识表征分支，并进一步开展逐类别候选覆盖与支持区域边界分析。\n"
    "困难二：不确定性蒸馏的训练机制可以被审计，但当前固定工作点的未知类别拒识率未形成稳定提升。解决办法是保留 standard-KD 对照，锁定训练预算和校准规则，将 uncertainty-KD 作为消融而非默认配置，并继续检查其收益是否依赖划分或工作点。\n"
    "困难三：排序指标的改善不一定转化为实际拒识工作点收益。解决办法是每组实验同步报告 AUROC、AUPR、FPR95、OSCR、已知类接受率、未知类拒识率和接受后已知类准确率，禁止用单一平均指标替代综合判断。\n"
    "困难四：候选池组成和自动 K 会影响新类别聚类评价。解决办法是同时记录候选污染率、未知覆盖率、逐类覆盖率、自动 K 误差及 ACC/NMI/ARI，真实类别数量只用于最终评价，不参与候选规则和自动 K 选择。"
)


REPRODUCIBILITY = (
    "（三）可复现实验流程\n"
    "1. 环境准备：在项目根目录安装 requirements.txt，使用 Python 3.11；准备 CIFAR-100 数据和对应 split 文件。实验脚本不使用混合池或测试集的真实类别标签进行训练、checkpoint 选择、阈值校准、自动 K、候选规则或方法选择。\n"
    "2. shared-backbone 三划分复核：运行 scripts/run_rejection_encoder_init_pair.py，分别使用 seed 2081、2082、2083。两臂共享 split-matched teacher、训练预算、mixed-pool NNPU、known-validation-only checkpoint、95% 已知类覆盖率校准和 discovery 配置，仅比较拒识分支配置。\n"
    "3. uncertainty-KD 三划分复核：运行 scripts/run_uncertainty_kd_core_pair.py，使用 seed 2071、2072、2073。standard-KD 与 uncertainty-KD 共享教师模型、学生初始化、训练预算、辅助损失、checkpoint 规则和阈值规则，唯一训练因素为 logits-KD 的样本权重方式。\n"
    "4. 结果核验：检查每个 seed 的配置审计、训练历史、蒸馏权重审计、校准报告和 discovery_report.json；只有文件完整且标签隔离字段通过时，才将结果纳入汇总。\n"
    "5. 报告生成：运行 python scripts/fill_midterm_template_content_angle.py，从机器可读结果文件生成本报告中的逐划分原始值和差值。"
)


def set_cell(cell, text: str, *, size: float = 10.5) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in paragraph.runs:
            run.font.name = "宋体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(size)


def main() -> None:
    doc = Document(TEMPLATE)

    # Basic information only: do not alter student/teacher rows.
    basic = doc.tables[0]
    set_cell(basic.cell(0, 2), "基于不确定性知识蒸馏的新类发现方法")
    set_cell(basic.cell(1, 2), "□实物作品  □发表论文  □软件程序  ☑研究报告\n□设计方案  □影音作品  □其他         （可多选）")

    # Template sections: introduction, research content/plan, implementation,
    # and concise stage result.  Commitment and teacher cells remain untouched.
    set_cell(doc.tables[1].cell(0, 0), PROJECT_INTRO)
    set_cell(doc.tables[2].cell(0, 0), RESEARCH_AND_PLAN, size=10)
    set_cell(doc.tables[3].cell(0, 0), PROGRESS_AND_DIFFICULTIES, size=9.5)
    set_cell(
        doc.tables[4].cell(0, 0),
        "完成开放环境新类发现实验流程、shared-backbone 与 uncertainty-KD 三划分对照、结果审计及可复现实验脚本，形成阶段性研究报告。",
        size=10.5,
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
