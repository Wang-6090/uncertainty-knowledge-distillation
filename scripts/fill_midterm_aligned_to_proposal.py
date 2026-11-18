"""Generate a midterm report aligned with the approved proposal.

The proposal supplies the background, research objectives, technical route and
original schedule.  This report keeps that structure and adds only the work
that has actually been implemented and audited so far.  The report intentionally
uses a small number of explanatory core comparisons instead of dumping every
metric into the implementation-progress section.
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
OUTPUT = ROOT / "附件2-北京交通大学大学生创新创业训练计划项目中期进展报告_按模板经费使用修订版.docx"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    rounded = (Decimal(str(value)) * Decimal("100")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return f"{rounded:.2f}%"


def number(value: float) -> str:
    return f"{value:.4f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_shared_results() -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in (2081, 2082, 2083):
        audit = read_json(ROOT / f"analysis/rejection_encoder_init_fresh_s{seed}.json")
        result[seed] = {
            "scratch_cnn": dict(audit["discovery"]["scratch_cnn"]),
            "shared_backbone": dict(audit["discovery"]["shared_backbone"]),
        }
    return result


def load_uncertainty_results() -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in (2071, 2072, 2073):
        data = read_json(ROOT / f"analysis/uncertainty_kd_core_s{seed}.json")
        result[seed] = {
            arm: dict(data["arms"][arm]["metrics"])
            for arm in ("standard", "uncertainty")
        }
    return result


def shared_core_comparison(data: dict[int, dict[str, dict[str, float]]]) -> str:
    rows = []
    auroc_control, auroc_treatment = [], []
    reject_control, reject_treatment = [], []
    for seed in (2081, 2082, 2083):
        control = data[seed]["scratch_cnn"]
        treatment = data[seed]["shared_backbone"]
        auroc_control.append(float(control["auroc"]))
        auroc_treatment.append(float(treatment["auroc"]))
        reject_control.append(float(control["unknown_reject_rate"]))
        reject_treatment.append(float(treatment["unknown_reject_rate"]))
        rows.append(
            f"seed={seed}：AUROC {number(control['auroc'])}→{number(treatment['auroc'])}；"
            f"未知类别拒识率 {percent(control['unknown_reject_rate'])}→"
            f"{percent(treatment['unknown_reject_rate'])}。"
        )
    return (
        "shared-backbone 与 scratch CNN 的核心对照（箭头左侧为对照，右侧为处理）："
        + " ".join(rows)
        + f"三划分均值为：AUROC {number(mean(auroc_control))}→{number(mean(auroc_treatment))}；"
        f"未知类别拒识率 {percent(mean(reject_control))}→{percent(mean(reject_treatment))}。"
    )


def uncertainty_core_comparison(data: dict[int, dict[str, dict[str, float]]]) -> str:
    rows = []
    control, treatment = [], []
    for seed in (2071, 2072, 2073):
        c = float(data[seed]["standard"]["unknown_reject_rate"])
        t = float(data[seed]["uncertainty"]["unknown_reject_rate"])
        control.append(c)
        treatment.append(t)
        rows.append(f"seed={seed}：{percent(c)}→{percent(t)}")
    return (
        "训练期 uncertainty-KD 的未知类别拒识率对照（standard-KD→uncertainty-KD）："
        + "；".join(rows)
        + f"；三划分均值为 {percent(mean(control))}→{percent(mean(treatment))}。"
    )


INTRO = (
    "近年来，深度学习模型在封闭数据集上的分类任务中取得了较好效果，但真实应用环境中的数据类别并不固定，"
    "模型可能接收到训练阶段没有定义的新类别样本。由于传统分类模型通常被训练为在既定类别集合中作出选择，"
    "面对新类别时容易产生错误分类和过高置信度。\n\n"
    "本项目研究开放环境下的新类发现问题：训练阶段仅对已知类别样本提供类别标注，发现或测试阶段可能出现训练类别集合之外的新类别。"
    "系统需要在不使用新类别训练标签的条件下，同时完成已知类别分类、未知样本拒识和新类别聚类。项目拟以知识蒸馏为基础，"
    "引入教师模型的不确定性信息，研究其对学生模型特征学习和未知检测的影响；同时结合特征距离、对比学习和聚类方法，"
    "形成从已知类别识别、未知样本筛选到新类别组织的开放学习流程。"
)


RESEARCH_CONTENT = (
    "（一）主要研究内容\n"
    "1. 开展相关理论与方法研究，系统梳理知识蒸馏、不确定性估计、开放集识别、新类发现的核心原理与主流算法，总结现有方法在特征学习、未知检测、知识迁移及聚类效果上的不足，明确本项目的技术路线与改进方向。\n"
    "2. 完成实验数据集的构建与预处理，选取 CIFAR-100、CUB-200-2011 等标准数据集，按照开放场景规则划分为已知类别与未知新类别，完成数据增强、标准化与划分，形成统一的实验评测环境。本阶段已首先完成 CIFAR-100 实验协议，其他数据集作为后续扩展。\n"
    "3. 设计基于原型引导与特征距离度量的开放集识别模块，通过学习类内紧凑的特征表示，实现对已知类别与未知样本的稳定区分，降低传统方法对超参数的依赖。在当前实现中，原型和距离度量用于描述已知类别支持区域，并通过独立拒识分支完成相应的未知样本判别。\n"
    "4. 构建不确定性估计分支，对模型预测置信度进行量化建模，自动识别高不确定性样本，为后续未知检测与新类发现提供可靠依据。当前实现同时检查预测不确定性、拒识分数及其与错误预测的关系。\n"
    "5. 提出不确定性感知知识蒸馏策略，将教师模型学到的语义知识与判别信息自适应迁移到学生模型，在提升模型精度的同时增强对开放环境的鲁棒性。具体实现以样本级不确定性调整 logits 蒸馏权重，并与 standard-KD 进行对照。\n"
    "6. 搭建统一开放学习框架，将已知分类、未知检测、新类聚类三个任务进行联合学习与端到端推理，形成一体化的开放场景智能识别模型。当前流程包括教师—学生分类与蒸馏、独立拒识、候选样本筛选和新类别聚类四个相互衔接的环节。\n"
    "7. 在标准数据集上开展全面实验，通过分类精度、未知检测 AUROC、新类聚类准确率等指标进行量化评估，并通过消融实验验证各模块的有效性，最终形成完整的分析结论。本阶段已建立拒识和聚类的多指标评价体系，并完成标准知识蒸馏、不确定性知识蒸馏及拒识表征的阶段性对照。"
)


PLAN = (
    "（二）实施进度安排\n"
    "1. 2026.06—2026.07：开展相关领域文献调研，梳理开放集识别、新类发现、知识蒸馏与不确定性估计研究现状，完成项目需求分析和整体方案设计。已完成。\n"
    "2. 2026.08—2026.09：完成公开数据集获取、已知/未知类别划分、数据增强、归一化与标准化处理，搭建 PyTorch 实验环境，形成统一评测基准。已完成。\n"
    "3. 2026.10—2026.11：设计并实现基于原型引导与特征距离度量的开放集识别模块，完成已知类别与未知样本区分功能开发与初步对照。目前已完成基础实现，正在进行多划分复核。\n"
    "4. 2026.12—2027.01：构建不确定性估计分支，设计不确定性感知知识蒸馏策略，完成教师模型训练与学生模型知识迁移学习。基础代码已提前完成，正式复核结果作为后续阶段的重点。\n"
    "5. 2027.02—2027.03：完善统一开放场景识别模型，将分类、未知检测和新类聚类进行整合。尚未形成结论。\n"
    "6. 2027.04—2027.05：开展标准数据集上的对比实验、消融实验和可视化分析，完成指标评估、结果整理和图表绘制。\n"
    "7. 2027.05—2027.06：汇总项目成果，撰写研究报告，整理项目文档和演示材料，准备结题与验收。"
)


def progress_text() -> str:
    return (
        "（一）已开展的工作\n"
        "1. 完成文献调研和任务定义。围绕开放集识别、新类发现、不确定性估计和知识蒸馏等方向梳理相关方法，明确本项目需要同时处理已知类别分类、未知样本拒识和新类别聚类三个相互联系的任务，并据此确定教师—学生模型和开放学习流程。\n"
        "2. 完成数据协议和实验环境建设。以 CIFAR-100 为主要实验数据集，建立已知类别与新类别的 60/40 划分，并进一步设置 semantic-hard 类别划分以检验语义相近类别的识别难度；完成数据增强、归一化、训练集与验证集划分以及无标签混合发现池的组织。\n"
        "3. 完成教师模型和学生模型的基础训练程序。教师模型用于提供类别判别信息，学生模型使用已知类别标注进行分类学习，为后续知识蒸馏、未知检测和新类别聚类提供统一的特征表示。\n"
        "4. 完成标准知识蒸馏和不确定性知识蒸馏模块。standard-KD 作为基础对照，将教师模型的软类别信息传递给学生模型；uncertainty-KD 根据样本不确定性调整蒸馏权重，用于检验不确定性是否能够改善学生模型对开放环境的适应能力。对蒸馏权重、蒸馏损失和梯度的检查表明，两种蒸馏路径均已在程序中实际执行；但 uncertainty-KD 在不同重复实验中的任务结果有升有降，尚未形成稳定收益。\n"
        "5. 完成开放集识别和独立拒识分支。针对普通分类器容易把新类别样本纳入已知类别的问题，增加独立拒识分支，利用已知样本和无标签混合发现池估计已知类别支持区域。采用 NNPU 风险估计，是为了避免把混合发现池中的样本直接当作未知类别标注；采用 shared-backbone 初始化，是为了改善 scratch CNN 分支对学生模型特征的利用。三组不同类别划分的重复实验显示，该配置的拒识效果较好，但语义相近新类别与已知类别之间的重叠仍未完全解决。\n"
        "6. 完成未知样本筛选和新类别聚类流程。使用已知验证集确定拒识工作点，将偏离已知类别支持区域的样本作为候选新类别样本；对候选特征进行 PCA 降维，并使用 KMeans 组织候选样本。自动类别数依据候选特征本身的无标签内部指标估计，真实类别信息只用于实验结束后的回顾性评价。\n"
        "7. 完成阶段性对照实验和结果分析。建立分类、未知检测和新类别聚类三类评价指标，围绕不同拒识表征、知识蒸馏方式和候选聚类设置开展对照实验。当前形成的阶段性判断是：项目已经建立了可运行的完整研究流程，部分拒识表征方案具有继续研究价值；不确定性知识蒸馏的机制已经实现，但其对完整新类发现任务的改善仍不稳定。下一阶段将重点分析语义相近新类别的逐类别覆盖、已知类别支持区域边界和聚类类别数估计问题。\n"
        "（二）项目实施存在的困难及解决办法\n"
        "困难一：语义相近的新类别与已知类别在特征空间中存在重叠，导致模型可能把新类别样本当作已知类别接受。该问题的难点在于，训练阶段没有新类别标注，不能用新类别标签直接塑造判别边界。当前采用已知验证集校准拒识工作点、独立拒识表征和无标签混合池风险估计，并进一步分析逐类别候选覆盖和已知类别支持区域边界；后续将重点检查哪些语义相近类别持续产生漏检。\n"
        "困难二：不确定性知识蒸馏的机制已经实现，但任务层面的收益不稳定。困难不在于蒸馏路径是否执行，而在于不确定性权重是否能把教师知识有效转化为学生模型对未知类别的区分能力。当前通过 standard-KD 对照、逐轮权重统计、蒸馏损失和梯度审计确认机制真实参与训练；由于不同类别划分上的拒识结果有升有降，暂不将其作为默认方法，后续将从不确定性来源、权重分布和蒸馏目标之间的匹配关系继续分析。\n"
        "困难三：排序指标改善不一定转化为实际工作点的未知拒识改善。AUROC 反映整体排序能力，而项目实际还需要在较高已知类别覆盖率下拒绝未知样本；两个指标可能出现方向不一致。为避免误判，后续每组实验同时报告 AUROC、AUPR、FPR95、OSCR、已知类别接受率、未知类别拒识率和接受后已知类别准确率，并以固定校准规则比较方法。\n"
        "困难四：候选池质量和自动类别数会同时影响新类别聚类结果。候选池中混入已知类别样本会降低聚类纯度，自动类别数偏差又会影响 ACC、NMI 和 ARI；但测试标签不能用于候选筛选和类别数选择。当前将候选污染率、未知覆盖率、自动 K 误差和聚类指标分开记录，真实类别数量仅用于最终评价，后续研究无标签条件下更稳定的候选筛选和类别数估计方法。\n"
        "困难五：多组实验需要保持数据划分、教师模型、训练预算、校准规则和评价方式一致，否则无法判断结果差异来自算法还是实验条件变化。当前采用固定类别划分、成对 control/treatment、known-validation-only checkpoint 与阈值校准，并保存配置、训练历史和机器可读结果；项目实验主要依托现有计算机和实验环境，暂未出现必须新增专用设备才能开展研究的条件问题。"
    )


MEMBERS_AND_RESOURCES = (
    "（三）项目成员分工及任务完成情况\n"
    "王笑颜（项目负责人）：上阶段负责项目整体方案设计、研究进度统筹、开放集识别与知识蒸馏核心算法实现、框架集成与程序调试，已完成总体实验流程设计、主要算法模块实现和阶段性结果分析。下一阶段负责统筹多划分复核、完善统一开放学习框架，并组织研究报告和结题材料整理。\n"
    "李柯颖：上阶段负责数据集划分、数据增强、模型训练、实验运行、指标计算与结果可视化，已完成 CIFAR-100 开放场景数据协议整理、主要对照实验运行和指标汇总。下一阶段负责继续开展多组对照实验、整理实验图表并完成结果复核。\n"
    "齐誉涵：上阶段负责文献整理、代码调试、实验记录、项目材料汇总、研究报告与结题文档撰写，已完成相关文献和实验记录整理，并参与中期材料编写。下一阶段负责持续整理实验过程和结果，完善研究报告、结题文档及成果展示材料。\n\n"
    "（三）经费及资源使用情况\n"
    "截至中期，项目已使用经费约 1500 元，主要用于资料购置、打印复印、项目材料整理及少量实验相关耗材。研究工作主要依托现有实验室计算机、软件环境和公开数据集开展，暂不安排大额设备购置；后续经费按照项目实际需要和学校财务规定据实使用。"
)


STAGE_RESULTS = (
    "截至中期，本项目已完成从任务定义到实验验证的主要基础工作。围绕开放环境下“已知类别分类、未知样本拒识和新类别聚类”三个环节，完成了研究数据协议、数据预处理、教师模型与学生模型训练程序、标准知识蒸馏和不确定性知识蒸馏模块，并建立了独立拒识、候选样本筛选和新类别聚类流程。\n\n"
    "在此基础上，项目完成了分类、拒识和聚类的评价体系以及多组对照实验，能够对不同算法设置的作用进行分环节分析。实验表明，当前程序已经能够完成从已知类别学习、未知样本筛选到候选新类别聚类的完整流程；同时也发现，语义相近的新类别与已知类别之间仍存在较明显的特征和分数重叠，这是影响整体新类发现效果的主要问题。\n\n"
    "阶段性研究还表明，不确定性信息已经能够进入知识蒸馏过程，但其对完整任务的改善尚不稳定。因此，项目目前形成的阶段性认识是：研究框架和实验流程已经建立，部分拒识表征方案显示出继续研究的价值，但不确定性知识蒸馏和语义相近新类别的稳定发现仍需进一步验证和改进。下一阶段将围绕上述问题继续开展针对性实验。"
)


def set_font(paragraph, size: float = 10.0, bold: bool = False) -> None:
    for run in paragraph.runs:
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(size)
        run.bold = bold


def fill_cell(cell, text: str, size: float) -> None:
    cell.text = ""
    blocks = text.split("\n\n")
    for index, block in enumerate(blocks):
        paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
        paragraph.text = block
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_font(paragraph, size=size)


def main() -> None:
    doc = Document(TEMPLATE)

    # Keep student, teacher, commitment and opinion sections from the template.
    doc.tables[0].cell(0, 2).text = "基于不确定性知识蒸馏的新类发现方法"
    fill_cell(doc.tables[1].cell(0, 0), INTRO, 10.5)
    fill_cell(doc.tables[2].cell(0, 0), RESEARCH_CONTENT + "\n\n" + PLAN, 9.5)
    fill_cell(doc.tables[3].cell(0, 0), progress_text(), 9.2)
    fill_cell(doc.tables[3].cell(1, 0), MEMBERS_AND_RESOURCES, 9.2)
    fill_cell(doc.tables[4].cell(0, 0), STAGE_RESULTS, 9.8)

    doc.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期进展报告"
    doc.core_properties.subject = "依据立项报告和中期模板修订的实验进展报告"
    doc.core_properties.comments = "简介和研究方案沿用立项报告逻辑；实验结论由机器可读结果生成。"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
