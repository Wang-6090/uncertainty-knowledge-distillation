"""Create a concise five-minute midterm presentation from the proposal deck."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法开题ppt.pptx"
OUTPUT = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法中期汇报.pptx"


def set_text(slide, index: int, text: str, *, size: int = 16, bold: bool = False) -> None:
    shape = slide.shapes[index]
    if not shape.has_text_frame:
        raise TypeError(f"shape {index} is not a text shape")
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    paragraphs = text.split("\n")
    for i, line in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.LEFT
        for run in p.runs:
            run.font.name = "微软雅黑"
            run.font.size = Pt(size)
            run.bold = bold


def set_title(slide, index: int, text: str) -> None:
    set_text(slide, index, text, size=22, bold=True)


def remove_slides_except(prs: Presentation, order: list[int]) -> None:
    ids = list(prs.slides._sldIdLst)
    selected = [ids[i] for i in order]
    for slide_id in ids:
        if slide_id not in selected:
            prs.part.drop_rel(slide_id.rId)
            prs.slides._sldIdLst.remove(slide_id)
    for slide_id in selected:
        if slide_id in prs.slides._sldIdLst:
            prs.slides._sldIdLst.remove(slide_id)
    for slide_id in selected:
        prs.slides._sldIdLst.append(slide_id)


def main() -> None:
    prs = Presentation(SOURCE)

    # Cover.
    set_text(prs.slides[0], 0, "基于不确定性知识蒸馏的新类发现方法", size=28, bold=True)
    set_text(prs.slides[0], 1, "MIDTERM PROGRESS REPORT", size=16, bold=False)
    set_text(prs.slides[0], 3, "中期汇报\n成员：王笑颜、李柯颖、齐誉涵", size=13)

    # Project background and research question.
    s = prs.slides[1]
    set_title(s, 6, "1. 项目背景与研究问题")
    set_text(
        s,
        11,
        "真实开放环境中的数据类别并不固定，模型可能接收到训练阶段没有定义的新类别。\n"
        "传统分类器会把未知样本强行归入已知类别，产生错误分类和过高置信度。\n"
        "本项目研究：在不使用新类别训练标签的条件下，同时完成已知类别分类、未知样本拒识和新类别聚类。",
        size=15,
    )
    set_text(s, 16, "项目目标", size=16, bold=True)
    set_text(s, 19, "研究价值", size=16, bold=True)
    set_text(
        s,
        20,
        "建立教师—学生知识蒸馏与开放集识别相结合的研究流程，分析不确定性信息对学生模型开放环境适应能力的影响。",
        size=12,
    )
    set_text(
        s,
        23,
        "形成从已知分类、未知筛选到新类聚类的可运行流程，为开放视觉感知任务提供实验依据。",
        size=12,
    )
    set_text(s, 21, "中期汇报重点：项目完成了哪些工作，当前证据说明了什么？", size=12, bold=True)
    set_text(s, 22, "研究内容与阶段进展", size=12, bold=True)

    # Research content, following the seven items in the proposal.
    s = prs.slides[4]
    set_title(s, 39, "2. 研究主要内容")
    set_text(s, 11, "理论与方法研究", size=13, bold=True)
    set_text(s, 30, "梳理知识蒸馏、不确定性估计、开放集识别和新类发现方法，明确技术路线。", size=10)
    set_text(s, 15, "数据集构建与预处理", size=13, bold=True)
    set_text(s, 31, "完成 CIFAR-100 开放场景划分、数据增强、标准化和无标签混合发现池组织。", size=10)
    set_text(s, 23, "开放集识别模块", size=13, bold=True)
    set_text(s, 32, "基于原型和特征距离描述已知支持区域，利用独立拒识分支识别偏离样本。", size=10)
    set_text(s, 27, "不确定性与知识蒸馏", size=13, bold=True)
    set_text(s, 33, "量化预测不确定性，并以样本级权重比较 standard-KD 与 uncertainty-KD。", size=10)
    set_text(s, 19, "统一框架与实验评价", size=13, bold=True)
    set_text(s, 34, "衔接分类、未知检测、候选筛选和新类聚类，使用多指标与消融实验进行评价。", size=10)

    # Technical route and actual pipeline.
    s = prs.slides[7]
    set_title(s, 0, "3. 技术路线与实验流程")
    set_text(s, 14, "理论与任务建模", size=13, bold=True)
    set_text(s, 17, "明确开放环境任务，梳理知识蒸馏、不确定性、开放集识别与新类发现的关系。", size=10)
    set_text(s, 15, "数据协议", size=13, bold=True)
    set_text(s, 22, "CIFAR-100 60/40 划分；semantic-hard 难例协议；已知标注样本 + 无标签混合发现池。", size=10)
    set_text(s, 16, "模型与训练", size=13, bold=True)
    set_text(s, 23, "教师—学生分类与蒸馏；standard-KD / uncertainty-KD；独立拒识分支与 NNPU。", size=10)
    set_text(s, 26, "统一推理", size=13, bold=True)
    set_text(s, 28, "已知类别分类 → 未知样本拒识 → 候选筛选 → 新类别聚类。", size=10)
    set_text(s, 31, "评价与对照", size=13, bold=True)
    set_text(s, 32, "AUROC、AUPR、FPR95、OSCR、未知拒识率、已知效用、ACC/NMI/ARI。", size=10)
    set_text(s, 33, "中期产出", size=13, bold=True)
    set_text(s, 36, "形成代码、数据划分、配置、训练记录、结果分析和中期研究材料。", size=10)

    # Implementation schedule and status.
    s = prs.slides[8]
    set_title(s, 0, "4. 实施进度与当前状态")
    schedule = [
        (6, 7, "文献调研、任务定义和总体方案", "已完成"),
        (9, 10, "数据集构建、类别划分和实验环境", "已完成"),
        (12, 13, "开放集识别模块与拒识分支", "已完成"),
        (21, 22, "不确定性估计与知识蒸馏", "已完成基础实现"),
        (25, 26, "统一框架复核与边界分析", "中期后重点"),
        (23, 24, "多数据集对比、消融和可视化", "后续开展"),
        (27, 28, "研究报告、论文和结题材料", "结题阶段"),
    ]
    for body_idx, date_idx, body, date in schedule:
        set_text(s, body_idx, body, size=9)
        set_text(s, date_idx, date, size=10, bold=True)

    # Stage results.
    s = prs.slides[9]
    set_title(s, 6, "5. 阶段性结果")
    set_text(s, 22, "整体完成情况", size=13, bold=True)
    set_text(s, 25, "已建立从已知类别学习、未知样本筛选到候选新类别聚类的完整实验流程，并形成配套评价体系。", size=11)
    set_text(s, 23, "shared-backbone 对照", size=13, bold=True)
    set_text(s, 26, "三组不同类别划分的均值：AUROC 0.7675→0.7995；未知类别拒识率 21.55%→25.16%。支持继续研究该拒识表征方案。", size=11)
    set_text(s, 24, "uncertainty-KD 对照", size=13, bold=True)
    set_text(s, 27, "权重、蒸馏损失和梯度真实生效，但不同重复实验结果有升有降，未知拒识均值 29.71%→26.39%，暂不作为默认配置。", size=11)

    # Difficulties and responses.
    s = prs.slides[3]
    set_title(s, 7, "6. 当前困难及解决办法")
    set_text(s, 13, "主要困难", size=15, bold=True)
    set_text(
        s,
        9,
        "• 语义相近新类别与已知类别在特征和拒识分数上重叠。\n"
        "• 排序指标的改善不一定转化为固定覆盖率下的未知拒识改善。\n"
        "• 候选池组成和自动类别数会影响新类聚类结果。\n"
        "• 不确定性蒸馏机制已实现，但任务层面的收益尚不稳定。",
        size=13,
    )
    set_text(s, 11, "解决方向", size=15, bold=True)
    set_text(
        s,
        12,
        "固定已知验证集校准规则，避免使用测试标签调阈值；\n"
        "分析逐类别候选覆盖和已知支持区域边界；\n"
        "联合报告排序、拒识、已知效用和聚类指标；\n"
        "继续比较不确定性来源、拒识表征和无标签候选筛选方法。",
        size=12,
    )

    # Summary.
    s = prs.slides[11]
    set_title(s, 0, "7. 中期总结")
    set_text(
        s,
        8,
        "• 已完成开题方案中主要实验基础、模型模块和评价流程的实现。\n\n"
        "• 当前形成的核心认识：研究框架已经建立，部分拒识表征方案具有继续研究价值；不确定性知识蒸馏已经真实进入训练，但整体任务收益尚未稳定。\n\n"
        "• 下一阶段重点：解决语义相近新类别的覆盖问题，完善支持区域边界分析，并继续验证统一开放学习流程。",
        size=17,
    )

    # Keep the closing slide and remove the proposal-only pages.
    remove_slides_except(prs, [0, 1, 4, 7, 8, 9, 3, 11, 12])
    prs.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期汇报"
    prs.core_properties.subject = "五分钟中期进展汇报"
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
