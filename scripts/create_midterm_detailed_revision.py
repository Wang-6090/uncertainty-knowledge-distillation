from __future__ import annotations

import importlib.util
from pathlib import Path
from shutil import copy2

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "基于不确定性知识蒸馏的新类发现方法中期进展报告_模型算法与困难详版.docx"
OUTPUT = ROOT / "基于不确定性知识蒸馏的新类发现方法中期进展报告_模型算法与困难详细扩展版.docx"


def set_cell_text(cell, text: str, size: float = 10.5) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "宋体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(size)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"未找到待扩展内容：{label}")
    return text.replace(old, new, 1)


def build_base() -> None:
    """复用已有详版生成逻辑，保证模板、格式和既有修订保持一致。"""
    script = ROOT / "scripts" / "create_midterm_algorithm_difficulties_revision.py"
    spec = importlib.util.spec_from_file_location("midterm_base_revision", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载中期报告基础修订脚本")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    if not BASE.exists():
        build_base()
    copy2(BASE, OUTPUT)
    doc = Document(str(OUTPUT))

    work_cell = doc.tables[3].cell(0, 0)
    work = work_cell.text
    work = replace_once(
        work,
        "2. 完成数据协议和实验环境建设。以 CIFAR-100 为主要实验数据集，建立已知类别与新类别的 60/40 划分，并进一步设置 semantic-hard 类别划分以检验语义相近类别的识别难度；完成数据增强、归一化、训练集与验证集划分以及无标签混合发现池的组织。",
        "2. 完成数据协议和实验环境建设。当前以 CIFAR-100 作为主要实验数据集，建立已知类别与未知类别的 60/40 划分，并进一步设置 semantic-hard 类别划分以检验语义相近类别的识别难度。选择 CIFAR-100 的原因是其类别数量适中、类别层次清晰，便于构造已知—未知类别协议并进行多组语义相近划分；当前阶段先在该数据集上完成流程闭环，后续再扩展到其他数据集。项目已完成数据增强、归一化、训练集与验证集划分，以及无标签混合发现池的组织，并固定了训练、验证、测试之间的数据隔离规则。",
        "数据协议扩展",
    )
    work = replace_once(
        work,
        "该结构为后续未知检测和新类别聚类提供统一的特征表示。",
        "训练阶段，分类头使用已知类别标签学习分类目标；拒识相关模块则接收主干特征并单独学习未知风险。这样既保留了已知类别分类能力，又为未知样本筛选提供了独立的评分路径。该结构为后续未知检测和新类别聚类提供统一的特征表示。",
        "模型训练目标扩展",
    )
    work = replace_once(
        work,
        "但不同重复实验中的任务结果有升有降，尚未形成稳定收益。",
        "在训练过程中，学生模型同时受到已知类别分类目标和蒸馏目标约束；uncertainty-KD 只改变逐样本蒸馏项的权重，不改变已知类别标签和基本分类任务。训练审计确认权重、蒸馏损失和相关参数梯度均实际参与优化，但不同重复实验中的任务结果有升有降，尚未形成稳定收益。",
        "蒸馏训练过程扩展",
    )
    work = replace_once(
        work,
        "三组不同类别划分的重复实验显示，该配置的拒识效果较好，但语义相近新类别与已知类别之间的重叠仍未完全解决。",
        "拒识流程按照“主干特征提取—拒识编码—风险估计—已知验证集校准—候选筛选”的顺序执行。三组不同类别划分的重复实验显示，shared-backbone 配置在当前拒识环节表现较好，但语义相近新类别与已知类别之间的重叠仍未完全解决；因此，拒识分数改善还需要结合候选覆盖率和已知污染率判断是否能够传递到后续聚类。",
        "拒识流程衔接扩展",
    )
    work = replace_once(
        work,
        "候选筛选和类别数选择不使用未知测试标签，真实类别信息只用于实验结束后的回顾性评价。",
        "候选筛选和类别数选择不使用未知测试标签，真实类别信息只用于实验结束后的回顾性评价。评价时先检查候选池是否覆盖了足够多的未知样本，并统计其中混入的已知样本比例；在候选池固定后，再使用 ACC、NMI 和 ARI 等指标分析聚类结果。通过分开记录候选池质量和聚类质量，可以判断误差主要来自前面的未知漏检、已知污染，还是来自后面的特征聚类。",
        "聚类评价扩展",
    )
    work = replace_once(
        work,
        "7. 完成阶段性对照实验和结果分析。建立分类、未知检测和新类别聚类三类评价指标，围绕不同拒识表征、知识蒸馏方式和候选聚类设置开展对照实验。",
        "7. 完成阶段性对照实验和结果分析。建立分类、未知检测和新类别聚类三类评价指标，围绕不同拒识表征、知识蒸馏方式和候选聚类设置开展对照实验。分类部分记录已知类别识别结果；未知检测部分同时记录 AUROC 和固定工作点下的未知类别拒识率；候选池部分记录未知覆盖率和已知污染率；聚类部分记录 ACC、NMI 和 ARI。这样的分层评价用于区分排序能力、实际拒识工作点、候选池质量和聚类质量，避免只依据单一指标判断方法是否有效。",
        "评价体系扩展",
    )
    set_cell_text(work_cell, work, size=10.2)

    difficulty_cell = doc.tables[3].cell(0, 0)
    difficulty = difficulty_cell.text
    difficulty += (
        "\n困难六：模型分支的作用边界需要在实验中保持清晰。教师不确定性、学生分类熵、学生不确定性头分数和拒识编码器输出都可以表现为数值分数，但它们产生于不同模块、服务于不同阶段，若在实验记录中混用，就无法判断收益来自蒸馏权重还是拒识评分。当前将教师不确定性权重、学生侧评分信号和拒识器输出分别保存，并在对照实验中单独打开或关闭相应开关；后续将继续检查不同信号的相关性及其对候选覆盖和污染的影响。"
    )
    set_cell_text(difficulty_cell, difficulty, size=10.2)

    result_cell = doc.tables[4].cell(0, 0)
    set_cell_text(
        result_cell,
        "截至中期，项目已完成开放环境下新类发现任务定义、CIFAR-100 类别划分与数据预处理，建立 ResNet-34 教师模型与 ResNet-18 学生模型，并完成分类头、不确定性估计头和独立拒识编码器的实现；明确 standard-KD 与 uncertainty-KD 的逐样本蒸馏差异，完成 NNPU 拒识、已知验证集校准、候选筛选、PCA 降维和 KMeans 聚类流程，形成分类、拒识、候选池和聚类评价体系。\n\n在 CIFAR-100 semantic-hard 60/40 协议下，三套预先固定的语义相近类别划分中，scratch CNN 基线的平均 AUROC 为 0.7675、未知类别拒识率为 21.55%；shared-backbone 配置的对应结果为 0.7995 和 25.16%，分别高出 0.0320 和 3.61 个百分点。该结果说明共享初始化在当前未知拒识环节具有相对一致的局部改善证据，但完整新类发现效果仍需结合候选覆盖、候选污染和聚类指标继续评价。uncertainty-KD 已确认真实参与训练，但跨划分任务收益尚未稳定。",
        size=10.2,
    )

    doc.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期进展报告（模型算法与困难详细扩展版）"
    doc.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
