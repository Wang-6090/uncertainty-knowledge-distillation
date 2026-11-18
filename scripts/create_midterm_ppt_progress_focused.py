"""Build a progress-focused five-minute midterm deck from the proposal theme."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法开题ppt.pptx"
OUTPUT = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法中期汇报_背景进展完善版.pptx"

NAVY = RGBColor(24, 54, 88)
TEMPLATE_NAVY = RGBColor(50, 74, 122)
TEAL = RGBColor(36, 139, 150)
ORANGE = RGBColor(235, 139, 54)
GREEN = RGBColor(48, 137, 91)
RED = RGBColor(187, 77, 77)
INK = RGBColor(36, 45, 58)
MUTED = RGBColor(95, 108, 124)
PALE = RGBColor(244, 247, 250)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(214, 224, 232)


def clear_all_slides(prs: Presentation) -> None:
    for slide in list(prs.slides):
        slide_id = slide.slide_id
        element = next(x for x in prs.slides._sldIdLst if int(x.id) == slide_id)
        prs.part.drop_rel(element.rId)
        prs.slides._sldIdLst.remove(element)


def set_run_font(run, size: int, color: RGBColor = INK, bold: bool = False) -> None:
    run.font.name = "微软雅黑"
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.bold = bold


def put_text(slide, x, y, w, h, text, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(2)
        run = p.add_run()
        run.text = line
        set_run_font(run, size, color, bold)
    return box


def rect(slide, x, y, w, h, fill=WHITE, line=LINE, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.8)
    return shape


def navigation(slide, section):
    """Recreate the proposal deck's original navigation treatment."""
    labels = ["项目介绍", "中期进展", "项目实施", "成果展示", "总结"]
    positions = [4.36, 6.15, 7.95, 9.74, 11.53]
    if section == "BACKGROUND & METHOD":
        active = 0
    elif section == "MIDTERM OVERVIEW":
        active = 1
    elif section in {"CURRENT PIPELINE", "WORK COMPLETED", "BOTTLENECK", "NEXT STEPS"}:
        active = 2
    elif section.startswith("EVIDENCE"):
        active = 3
    else:
        active = 4
    for i, (x, label) in enumerate(zip(positions, labels)):
        if i == active:
            rect(slide, x, 0.03, 1.79, 0.60, TEMPLATE_NAVY, TEMPLATE_NAVY, True)
        put_text(slide, x + 0.03, 0.10, 1.73, 0.47, label, 20, WHITE if i == active else RGBColor(191, 191, 191), True, PP_ALIGN.CENTER)
    rect(slide, 0, 0.65, 13.33, 0.01, TEMPLATE_NAVY, TEMPLATE_NAVY)


def title(slide, section, text):
    # Reuse the proposal's original top navigation and left-aligned title area.
    navigation(slide, section)
    put_text(slide, 0.51, 1.10, 6.94, 0.74, text, 24, TEMPLATE_NAVY, True)


def footer(slide, number):
    put_text(slide, 0.58, 7.05, 4.4, 0.2, "基于不确定性知识蒸馏的新类发现方法｜中期汇报", 8, MUTED)
    put_text(slide, 12.15, 7.05, 0.55, 0.2, str(number), 8, MUTED, False, PP_ALIGN.RIGHT)


def content_slide(prs: Presentation, section, heading, number):
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    title(slide, section, heading)
    footer(slide, number)
    return slide


def shift_body_content(prs: Presentation, shift_inches: float = 0.5) -> None:
    """Leave the original title band untouched and move body content below it."""
    shift = Inches(shift_inches)
    for slide in list(prs.slides)[1:]:
        for shape in slide.shapes:
            top = shape.top / 914400
            if 1.30 <= top < 6.90:
                shape.top += shift


def card(slide, x, y, w, h, heading, body, accent=TEAL, body_size=12):
    rect(slide, x, y, w, h, WHITE, LINE, True)
    rect(slide, x, y, 0.08, h, accent, accent)
    put_text(slide, x + 0.22, y + 0.14, w - 0.38, 0.38, heading, 16, accent, True)
    put_text(slide, x + 0.22, y + 0.58, w - 0.38, h - 0.7, body, body_size, INK)


def table(slide, x, y, w, h, headers, rows, widths=None, font_size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if widths:
        for i, width in enumerate(widths):
            tbl.columns[i].width = Inches(width)
    for c, value in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = value
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                set_run_font(run, font_size, WHITE, True)
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r % 2 else WHITE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
                for run in p.runs:
                    set_run_font(run, font_size, INK, c == 0)
    return tbl


def main() -> None:
    prs = Presentation(SOURCE)
    clear_all_slides(prs)

    # 1. Cover.
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    rect(slide, 0, 0, 13.33, 7.5, PALE, PALE)
    rect(slide, 0, 0, 13.33, 1.05, NAVY, NAVY)
    rect(slide, 0, 6.62, 13.33, 0.88, NAVY, NAVY)
    put_text(slide, 0.85, 1.72, 11.7, 0.72, "基于不确定性知识蒸馏的新类发现方法", 38, NAVY, True, PP_ALIGN.CENTER)
    put_text(slide, 1.6, 2.55, 10.1, 0.34, "UNCERTAINTY-AWARE KNOWLEDGE DISTILLATION FOR NOVEL CLASS DISCOVERY", 11, TEAL, False, PP_ALIGN.CENTER)
    rect(slide, 1.42, 3.35, 10.5, 1.16, WHITE, LINE, True)
    put_text(slide, 1.75, 3.62, 9.85, 0.62, "中期进展汇报\n从方案设计走向实验验证", 21, INK, True, PP_ALIGN.CENTER)
    put_text(slide, 1.0, 6.9, 11.3, 0.22, "项目负责人：王笑颜　项目成员：李柯颖、齐誉涵", 11, WHITE, False, PP_ALIGN.CENTER)

    # 2. Background and overall method.
    slide = content_slide(prs, "BACKGROUND & METHOD", "1.1 研究背景与总体方法", 2)
    put_text(slide, 0.78, 1.42, 11.8, 0.56, "开放环境下，训练阶段只能获得已知类别标注，而测试样本可能来自未参与训练的新类别。因此，系统不仅要完成已知类别识别，还要识别未知样本并进一步组织新类别。", 13, MUTED)
    card(slide, 0.78, 2.22, 3.75, 3.15, "研究背景", "现实场景中的类别集合并不封闭。\n\n如果模型只优化已知类别分类准确率，未知样本可能被错误地接受为已知类别，后续新类发现也会受到影响。", TEAL, 13)
    card(slide, 4.78, 2.22, 3.75, 3.15, "研究目标", "在不使用新类别训练标签的条件下，建立“已知分类—未知检测—新类聚类”的一体化实验流程。\n\n重点考察未知检测和新类发现能力。", ORANGE, 13)
    card(slide, 8.78, 2.22, 3.75, 3.15, "总体方法", "以教师—学生知识蒸馏学习分类表征；引入不确定性估计调节知识迁移；再通过拒识分支筛选候选样本，并对候选特征进行聚类。", NAVY, 13)
    rect(slide, 0.78, 5.72, 11.75, 0.62, PALE, LINE, True)
    put_text(slide, 1.0, 5.86, 11.3, 0.3, "本阶段重点：先验证流程和模块是否真实有效，再判断各模块能否稳定改善完整的新类发现任务。", 12, INK, True, PP_ALIGN.CENTER)

    # 3. Progress at a glance.
    slide = content_slide(prs, "MIDTERM OVERVIEW", "2.1 中期进展主线：从方案设计到问题定位", 3)
    put_text(slide, 0.78, 1.42, 11.8, 0.35, "截至中期，研究已完成从实验协议建立、基线复现到关键模块对照的第一轮验证，当前重点转向解释差异并定位瓶颈。", 14, MUTED)
    card(slide, 0.78, 2.05, 3.85, 3.5, "阶段一｜建立实验基础", "明确开放环境下的已知分类、未知检测和新类聚类任务。\n\n完成类别划分、数据预处理、训练协议和评价指标，形成可重复运行的实验入口。", TEAL, 13)
    card(slide, 4.75, 2.05, 3.85, 3.5, "阶段二｜完成模块对照", "在统一协议下完成 scratch CNN、shared-backbone、standard-KD 和 uncertainty-KD 的对照。\n\n同步检查蒸馏损失、权重和梯度，确认方法确实进入训练。", GREEN, 13)
    card(slide, 8.72, 2.05, 3.85, 3.5, "阶段三｜形成阶段判断", "shared-backbone 在三组类别划分中改善了拒识指标。\n\nuncertainty-KD 已实现，但任务层面的收益不稳定；当前瓶颈集中在语义相近类别的表征重叠。", RED, 13)

    # 4. Current pipeline.
    slide = content_slide(prs, "CURRENT PIPELINE", "3.1 当前实验流程：每一轮实验如何推进", 4)
    steps = [
        ("阶段 1\n数据与基线", "已知类别训练\n建立基础模型", TEAL),
        ("阶段 2\n教师—学生训练", "standard-KD /\nuncertainty-KD", ORANGE),
        ("阶段 3\n未知检测", "独立拒识分支\n+ NNPU", RED),
        ("阶段 4\n候选筛选", "已知验证集\n校准工作点", GREEN),
        ("阶段 5\n新类发现评价", "PCA + KMeans\n多指标复核", NAVY),
    ]
    x0 = 0.55
    for i, (head, body, color) in enumerate(steps):
        x = x0 + i * 2.55
        rect(slide, x, 2.05, 2.05, 2.0, WHITE, color, True)
        put_text(slide, x + 0.12, 2.28, 1.82, 0.55, head, 15, color, True, PP_ALIGN.CENTER)
        put_text(slide, x + 0.17, 3.05, 1.72, 0.62, body, 11, INK, False, PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            put_text(slide, x + 2.12, 2.7, 0.4, 0.35, "→", 22, MUTED, True, PP_ALIGN.CENTER)
    rect(slide, 0.78, 4.85, 11.8, 1.0, PALE, LINE, True)
    put_text(slide, 1.05, 5.08, 11.2, 0.5, "条件控制原则：训练、模型选择、阈值校准和自动 K 选择只使用训练集、已知验证集或无标签内部信息；未知类别标签仅用于最终评价和误差分析。", 12, INK, False, PP_ALIGN.CENTER)

    # 5. Work completed.
    slide = content_slide(prs, "WORK COMPLETED", "3.2 已开展工作：完成第一轮可运行实验", 5)
    card(slide, 0.78, 1.55, 3.75, 1.95, "01｜数据与实验协议", "梳理开放集识别、新类发现、不确定性估计和知识蒸馏方法；完成 CIFAR-100 60/40 与 semantic-hard 类别划分和预处理。", TEAL, 12)
    card(slide, 4.78, 1.55, 3.75, 1.95, "02｜基线与蒸馏模型", "建立 scratch CNN、教师模型和学生模型，完成 standard-KD 与 uncertainty-KD 的统一训练入口。", ORANGE, 12)
    card(slide, 8.78, 1.55, 3.75, 1.95, "03｜训练过程审计", "检查不确定性权重、蒸馏损失和梯度是否真实进入优化，避免只改变配置而未改变训练过程。", RED, 12)
    card(slide, 0.78, 4.0, 3.75, 1.95, "04｜未知检测与拒识", "完成独立拒识分支和 NNPU；拒识工作点只由已知验证集校准，避免使用未知标签调参。", GREEN, 12)
    card(slide, 4.78, 4.0, 3.75, 1.95, "05｜候选筛选与聚类", "将偏离已知支持区域的样本作为候选，使用 PCA 和 KMeans 进行新类聚类，并用无标签内部指标估计类别数。", NAVY, 12)
    card(slide, 8.78, 4.0, 3.75, 1.95, "06｜对照与重复验证", "围绕 shared-backbone 和 uncertainty-KD 开展成对对照，在不同类别划分下比较拒识、已知效用和聚类表现。", TEAL, 12)

    # 6. Evidence: shared backbone.
    slide = content_slide(prs, "EVIDENCE 1", "4.1 阶段性证据一：shared-backbone 改善了拒识表征", 6)
    put_text(slide, 0.8, 1.38, 11.8, 0.35, "在完成基线后，首先检验共享主干是否能改善拒识分支所使用的特征表示；三组实验仅改变类别划分，其他主要训练和校准条件保持一致。", 12, MUTED)
    table(slide, 0.85, 2.0, 8.0, 2.75, ["指标", "对照", "shared-backbone", "变化"], [["AUROC（均值）", "0.7675", "0.7995", "+0.0320"], ["未知类别拒识率（均值）", "21.55%", "25.16%", "+3.61 个百分点"], ["三组划分方向", "—", "均为改善", "3/3"]], [2.25, 1.9, 2.1, 1.75], 12)
    card(slide, 9.2, 2.0, 3.35, 2.75, "阶段性解释", "shared-backbone 使拒识分支能够更充分利用学生表征，说明该配置具有进一步研究价值。\n\n这项结果只支持拒识表征的局部改善，不能直接推断完整新类发现任务已经解决。", GREEN, 12)
    rect(slide, 0.85, 5.15, 11.7, 0.9, PALE, LINE, True)
    put_text(slide, 1.05, 5.38, 11.3, 0.42, "下一步：检查逐类别候选覆盖和已知支持区域边界，确认改善是否真正覆盖 semantic-near hard unknown。", 13, INK, True, PP_ALIGN.CENTER)

    # 7. Evidence: uncertainty KD.
    slide = content_slide(prs, "EVIDENCE 2", "4.2 阶段性证据二：uncertainty-KD 已实现，但收益不稳定", 7)
    put_text(slide, 0.8, 1.38, 11.8, 0.35, "在拒识表征对照之后，进一步检验不确定性感知蒸馏是否能带来稳定收益；训练审计确认相关权重、损失和梯度均已进入优化过程。", 12, MUTED)
    table(slide, 0.85, 1.95, 8.0, 3.25, ["重复实验", "standard-KD", "uncertainty-KD", "变化"], [["划分 A", "29.63%", "22.90%", "−6.73 pp"], ["划分 B", "30.55%", "26.80%", "−3.75 pp"], ["划分 C", "28.95%", "29.48%", "+0.53 pp"], ["均值", "29.71%", "26.39%", "−3.32 pp"]], [2.0, 1.9, 2.0, 2.1], 12)
    card(slide, 9.2, 1.95, 3.35, 3.25, "阶段性解释", "训练机制：已真实执行。\n\n任务结果：三组实验中两组下降、一组略有上升，尚未形成稳定收益。\n\n当前决策：保留为消融方案，不作为默认训练方法。", RED, 12)
    rect(slide, 0.85, 5.55, 11.7, 0.62, PALE, LINE, True)
    put_text(slide, 1.05, 5.7, 11.3, 0.3, "后续不再根据单一有利指标继续调权重，优先分析不确定性来源与任务目标是否匹配。", 12, INK, True, PP_ALIGN.CENTER)

    # 8. Bottleneck diagnosis.
    slide = content_slide(prs, "BOTTLENECK", "4.3 当前问题定位：流程已经跑通，但整体收益仍不稳定", 8)
    card(slide, 0.78, 1.55, 3.75, 3.85, "问题 1｜表征重叠", "语义相近的新类别与已知类别在特征空间和拒识分数上重叠。\n\n结果：未知样本容易被接受为已知类别，候选池覆盖不足。\n\n应对：逐类别覆盖审计、支持区域边界分析。", RED, 12)
    card(slide, 4.78, 1.55, 3.75, 3.85, "问题 2｜指标不一致", "AUROC 反映整体排序，实际系统还要在高已知覆盖率工作点拒绝未知。\n\n结果：排序改善不一定转化为固定工作点收益。\n\n应对：联合报告排序、拒识和已知效用。", ORANGE, 12)
    card(slide, 8.78, 1.55, 3.75, 3.85, "问题 3｜聚类不稳定", "候选池污染和自动 K 偏差会同时影响新类聚类。\n\n结果：聚类指标可能改善，但类别数估计仍不准确。\n\n应对：无标签候选筛选与 K 估计分开验证。", NAVY, 12)
    put_text(slide, 0.9, 5.85, 11.6, 0.42, "因此，下一阶段不再简单增加模块，而是沿着“表征—拒识—候选—聚类”链路定位语义相近新类别的具体漏检环节。", 14, NAVY, True, PP_ALIGN.CENTER)

    # 9. Next steps.
    slide = content_slide(prs, "NEXT STEPS", "4.4 下一阶段计划：围绕核心瓶颈开展可验证复核", 9)
    card(slide, 0.78, 1.55, 3.75, 3.8, "第一步｜覆盖诊断", "固定已知类别覆盖率和候选规则，统计逐类别未知覆盖、候选污染和漏检样本。\n\n目标：找出 semantic-near hard unknown 的具体漏检类别。", TEAL, 12)
    card(slide, 4.78, 1.55, 3.75, 3.8, "第二步｜表征改进", "在新划分上比较 shared-backbone、容量、初始化和支持区域目标，保持训练预算与校准规则一致。\n\n目标：验证拒识改善是否跨划分、跨类别稳定。", GREEN, 12)
    card(slide, 8.78, 1.55, 3.75, 3.8, "第三步｜蒸馏复核", "固定 standard-KD 对照，分析不确定性来源、权重分布和蒸馏目标的匹配关系。\n\n目标：只有在拒识、已知效用和聚类联合改善时才考虑晋升默认。", ORANGE, 12)
    rect(slide, 0.78, 5.78, 11.75, 0.62, PALE, LINE, True)
    put_text(slide, 1.0, 5.92, 11.3, 0.3, "后续仍坚持：新划分预先固定、成对对照、仅用已知验证集校准，测试标签只用于最终评价。", 12, INK, True, PP_ALIGN.CENTER)

    # 10. Conclusion.
    slide = content_slide(prs, "CONCLUSION", "5.1 中期结论", 10)
    rect(slide, 0.85, 1.55, 11.7, 3.8, PALE, LINE, True)
    put_text(slide, 1.2, 1.9, 11.0, 0.55, "已经完成", 18, TEAL, True, PP_ALIGN.CENTER)
    put_text(slide, 1.35, 2.48, 10.7, 0.72, "从任务定义、数据协议、模型模块到未知检测和新类聚类的完整实验流程。", 17, INK, False, PP_ALIGN.CENTER)
    put_text(slide, 1.2, 3.35, 11.0, 0.55, "阶段性认识", 18, ORANGE, True, PP_ALIGN.CENTER)
    put_text(slide, 1.35, 3.92, 10.7, 0.72, "shared-backbone 有明确局部证据；uncertainty-KD 尚未形成稳定整体收益；核心瓶颈是语义相近类别重叠。", 17, INK, False, PP_ALIGN.CENTER)
    put_text(slide, 1.2, 4.78, 11.0, 0.55, "后续重点", 18, NAVY, True, PP_ALIGN.CENTER)
    put_text(slide, 1.35, 5.35, 10.7, 0.46, "围绕逐类别覆盖、支持区域边界和无标签候选机制继续验证。", 16, INK, False, PP_ALIGN.CENTER)

    shift_body_content(prs)

    prs.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期汇报（背景与进展完善版）"
    prs.core_properties.subject = "五分钟中期汇报"
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
