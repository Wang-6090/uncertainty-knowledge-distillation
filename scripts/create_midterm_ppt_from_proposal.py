"""Create the midterm deck by editing the proposal deck in place.

The proposal file is used as the visual master: its slides, navigation,
shapes, spacing, colors, and original layout are preserved. Only the text
content is replaced with midterm-specific material.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法开题ppt.pptx"
OUTPUT = ROOT / "24301044_王笑颜_基于不确定性知识蒸馏的新类发现方法中期汇报_研究背景页调整版.pptx"

INK = RGBColor(36, 45, 58)
TEMPLATE_NAVY = RGBColor(50, 74, 122)
MUTED = RGBColor(95, 108, 124)


def slide_shapes(slide):
    # Iteration is more reliable than slicing for this particular source file.
    return [shape for shape in slide.shapes]


def set_shape_text(slide, index: int, text: str, size: int | None = 14,
                   color: RGBColor | None = INK, bold: bool | None = False,
                   align=None) -> None:
    shape = slide_shapes(slide)[index]
    if not hasattr(shape, "text_frame"):
        raise TypeError(f"shape {index} is not a text shape")
    tf = shape.text_frame
    sample_run = next((run for paragraph in tf.paragraphs for run in paragraph.runs if run.text.strip()), None)
    sample_alignment = tf.paragraphs[0].alignment if tf.paragraphs else None
    tf.clear()
    tf.word_wrap = True
    # The existing text frame is the style source. This preserves the original
    # template's theme font, size, color, boldness, margins, and alignment.
    if sample_run is not None:
        sample_name = sample_run.font.name
        sample_size = sample_run.font.size.pt if sample_run.font.size else None
        sample_bold = sample_run.font.bold
        sample_color_type = sample_run.font.color.type
        sample_color = sample_run.font.color.rgb if sample_color_type == 1 else None
        if sample_name:
            font_name = sample_name
        else:
            font_name = None
        if sample_size is not None:
            font_size = sample_size
        else:
            font_size = None
        if sample_bold is not None:
            font_bold = sample_bold
        else:
            font_bold = None
        font_color = sample_color
        paragraph_alignment = sample_alignment if sample_alignment is not None else align
    else:
        font_name = "微软雅黑"
        font_size = size
        font_bold = bold
        font_color = color
        paragraph_alignment = align
    for line_index, line in enumerate(text.split("\n")):
        paragraph = tf.paragraphs[0] if line_index == 0 else tf.add_paragraph()
        if paragraph_alignment is not None:
            paragraph.alignment = paragraph_alignment
        run = paragraph.add_run()
        run.text = line
        if font_name:
            run.font.name = font_name
        if font_size is not None:
            run.font.size = Pt(font_size)
        if font_color is not None:
            run.font.color.rgb = font_color
        if font_bold is not None:
            run.bold = font_bold


def set_title(slide, index: int, text: str) -> None:
    set_shape_text(slide, index, text, 24, TEMPLATE_NAVY, True)


def set_nav_text(slide, index: int, text: str) -> None:
    set_shape_text(slide, index, text, 20, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)


def set_runs_style(slide, index: int, color: RGBColor | None = None,
                   size: int | None = None, bold: bool | None = None) -> None:
    """Apply an explicit correction when the inherited template color is unsuitable."""
    shape = slide_shapes(slide)[index]
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            if color is not None:
                run.font.color.rgb = color
            if size is not None:
                run.font.size = Pt(size)
            if bold is not None:
                run.bold = bold


def set_nav_state(slide, active: int) -> None:
    """Set the active top-navigation tab while retaining the proposal geometry."""
    positions = [4.36, 6.15, 7.95, 9.74, 11.53]
    shapes = slide_shapes(slide)
    background = next(
        (shape for shape in shapes
         if not (hasattr(shape, "text") and shape.text.strip())
         and 0.0 <= shape.top / 914400 <= 0.1
         and 1.7 <= shape.width / 914400 <= 1.9
         and 0.5 <= shape.height / 914400 <= 0.7),
        None,
    )
    if background is not None:
        background.left = Inches(positions[active])
        background.fill.solid()
        background.fill.fore_color.rgb = TEMPLATE_NAVY
        background.line.fill.background()
    for shape in shapes:
        if not (hasattr(shape, "text_frame") and shape.text.strip()):
            continue
        if shape.top is None or shape.left is None or shape.width is None:
            continue
        top = shape.top / 914400
        left = shape.left / 914400
        width = shape.width / 914400
        if top < 0.65 and 1.6 <= width <= 1.9:
            nearest = min(range(5), key=lambda i: abs(left - (positions[i] + 0.03)))
            if abs(left - (positions[nearest] + 0.03)) < 0.12:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor(255, 255, 255) if nearest == active else RGBColor(191, 191, 191)
                        run.bold = True


def resize_shape(slide, index: int, x: float | None = None, y: float | None = None,
                 w: float | None = None, h: float | None = None) -> None:
    shape = slide_shapes(slide)[index]
    if x is not None:
        shape.left = Inches(x)
    if y is not None:
        shape.top = Inches(y)
    if w is not None:
        shape.width = Inches(w)
    if h is not None:
        shape.height = Inches(h)


def remove_shapes(slide, indices: tuple[int, ...]) -> None:
    shapes = slide_shapes(slide)
    for index in sorted(indices, reverse=True):
        shape = shapes[index]
        shape._element.getparent().remove(shape._element)


def set_new_text(shape, text: str, size: int, color: RGBColor,
                 bold: bool = False, align=PP_ALIGN.LEFT) -> None:
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for line_index, line in enumerate(text.split("\n")):
        paragraph = tf.paragraphs[0] if line_index == 0 else tf.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(2)
        run = paragraph.add_run()
        run.text = line
        run.font.name = "微软雅黑"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.bold = bold


def add_work_card(slide, x: float, y: float, w: float, h: float,
                  number: str, heading: str, body: str) -> None:
    """Add one equally sized work item to the completed-work page."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(247, 249, 253)
    card.line.color.rgb = RGBColor(196, 209, 230)
    card.line.width = Pt(1.1)

    badge = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(x + 0.24), Inches(y + 0.24), Inches(0.42), Inches(0.42)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = TEMPLATE_NAVY
    badge.line.fill.background()
    badge_tf = badge.text_frame
    badge_tf.clear()
    badge_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = badge_tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = number
    run.font.name = "微软雅黑"
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

    title = slide.shapes.add_textbox(
        Inches(x + 0.82), Inches(y + 0.20), Inches(w - 1.05), Inches(0.48)
    )
    set_new_text(title, heading, 16, TEMPLATE_NAVY, True)

    body_box = slide.shapes.add_textbox(
        Inches(x + 0.28), Inches(y + 0.82), Inches(w - 0.56), Inches(h - 1.05)
    )
    set_new_text(body_box, body, 12, RGBColor(51, 51, 51), False)
    body_box.text_frame.margin_left = Inches(0.02)
    body_box.text_frame.margin_right = Inches(0.02)
    body_box.text_frame.margin_top = Inches(0.02)
    body_box.text_frame.margin_bottom = Inches(0.02)


def add_visual_table(slide, x: float, y: float, w: float, h: float,
                     rows: list[list[str]], font_size: int = 9) -> None:
    """Add a compact, template-colored table for quantitative results."""
    table_shape = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)
    )
    table = table_shape.table
    for col in range(len(rows[0])):
        table.columns[col].width = Inches(w / len(rows[0]))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = TEMPLATE_NAVY if row_index == 0 else RGBColor(247, 249, 253)
            cell.margin_left = Inches(0.03)
            cell.margin_right = Inches(0.03)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.name = "微软雅黑"
                    run.font.size = Pt(font_size)
                    run.font.bold = row_index == 0
                    run.font.color.rgb = RGBColor(255, 255, 255) if row_index == 0 else RGBColor(51, 51, 51)


def copy_template_slide(prs, template_slide):
    """Copy the template slide's native shapes into a new slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    for shape in template_slide.shapes:
        slide.shapes._spTree.insert_element_before(deepcopy(shape._element), "p:extLst")
    return slide


def move_slide_to(prs, slide, position: int) -> None:
    slide_ids = prs.slides._sldIdLst
    element = next(item for item in slide_ids if int(item.id) == slide.slide_id)
    slide_ids.remove(element)
    slide_ids.insert(position, element)


def main() -> None:
    prs = Presentation(SOURCE)
    template_prs = Presentation(SOURCE)
    # The proposal template contains a complete section 2. Reuse its native
    # layouts so the midterm deck keeps the original chapter structure.
    section_21 = copy_template_slide(prs, template_prs.slides[4])
    section_22 = copy_template_slide(prs, template_prs.slides[5])

    # 1. Cover: retain the original cover graphics and replace only the subtitle.
    set_shape_text(prs.slides[0], 1, "MIDTERM PROGRESS REPORT", 20, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)

    # 2. Background and research objective.
    slide = prs.slides[1]
    set_title(slide, 6, "1.1 研究背景与研究目标")
    set_shape_text(
        slide, 11,
        "封闭集模型默认训练类别与测试类别一致，面对训练阶段未出现的新类别时，容易将未知样本高置信度地误判为已知类别。\n"
        "开放环境下的系统需要先正确识别已知类别，再拒识未知样本，并在没有新类标注的条件下组织潜在新类别。",
        14,
    )
    set_shape_text(slide, 21, "研究背景：", 14, TEMPLATE_NAVY, True)
    set_shape_text(slide, 19, "总体目标", 14, TEMPLATE_NAVY, True)
    set_shape_text(
        slide, 20,
        "围绕不确定性估计与知识蒸馏，构建分类、未知检测和候选聚类相衔接的实验框架，分析不同表征与蒸馏策略对开放环境识别的作用。",
        14,
    )
    set_shape_text(slide, 16, "中期目标", 14, TEMPLATE_NAVY, True)
    set_shape_text(slide, 22, "研究目标：", 14, TEMPLATE_NAVY, True)
    set_shape_text(
        slide, 23,
        "完成统一数据协议、基线与对照模型、拒识和候选聚类流程，建立多指标评价方法，明确已经验证的局部收益及仍待解决的任务瓶颈。",
        14,
    )

    # 3. Method overview.
    slide = prs.slides[2]
    set_title(slide, 0, "1.2 总体研究方法")
    set_shape_text(slide, 17, "开放环境任务定义", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 16, "训练阶段只使用已知类别标签；测试阶段将未知样本纳入拒识评价，不把未知标签用于训练或阈值校准。", 14)
    set_shape_text(slide, 20, "教师—学生知识蒸馏", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 19, "教师模型提供分类语义和判别信息，学生模型通过 standard-KD 或 uncertainty-KD 学习可迁移表征。", 14)
    set_shape_text(slide, 23, "未知候选与新类聚类", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 22, "拒识分支筛选偏离已知支持区域的候选，再使用 PCA 与 KMeans 组织潜在新类别。", 14)
    set_shape_text(slide, 26, "不确定性估计", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 25, "量化预测不确定性，识别高不确定性样本，并检验不确定性感知蒸馏是否改善未知检测。", 14)
    for heading_index in (17, 20, 23, 26):
        set_runs_style(slide, heading_index, TEMPLATE_NAVY, bold=True)
    for body_index in (16, 19, 22, 25):
        set_runs_style(slide, body_index, RGBColor(51, 51, 51))

    # 4. Research content (section 2.1 from the proposal structure).
    slide = section_21
    set_nav_state(slide, 1)
    set_title(slide, 39, "2.1 项目研究内容")
    set_shape_text(slide, 11, "数据协议与实验环境", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 30, "围绕 CIFAR-100 开放类别划分，统一数据预处理、数据增强以及训练/验证/测试隔离，建立可比较的实验条件。", 14)
    set_shape_text(slide, 15, "开放集识别与拒识", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 31, "利用分类表征和拒识分支筛选偏离已知类别支持区域的样本，降低未知样本被误判为已知类别的风险。", 14)
    set_shape_text(slide, 23, "不确定性估计", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 32, "量化模型预测的不确定性，为未知样本筛选和蒸馏权重调整提供依据。", 14)
    set_shape_text(slide, 27, "不确定性感知知识蒸馏", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 33, "根据样本不确定性调整知识迁移强度，研究其对开放环境特征学习和未知检测的影响。", 14)
    set_shape_text(slide, 19, "统一框架与实验验证", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 34, "将分类、拒识、候选筛选、新类聚类和多指标评价串联，分析各模块的实际作用。", 14)
    for heading_index in (11, 15, 19, 23, 27):
        set_runs_style(slide, heading_index, RGBColor(255, 255, 255), bold=True)

    # 5. Method improvements and research focus (section 2.2).
    slide = section_22
    set_nav_state(slide, 1)
    set_title(slide, 0, "2.2 方法改进与研究重点")
    set_shape_text(slide, 11, "统一任务链路", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 9, "将已知类别分类、未知检测和新类聚类纳入同一评价流程，避免只依据单一拒识指标判断方法效果。", 14)
    set_shape_text(slide, 17, "共享表征对照", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 15, "比较 scratch CNN、standard-KD 与 shared-backbone，检验共享表征对未知检测的实际作用。", 14)
    set_shape_text(slide, 23, "不确定性感知蒸馏", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 21, "审计不确定性权重、蒸馏损失和梯度，确认该机制是否真正进入训练，并评估其收益稳定性。", 14)
    set_shape_text(slide, 29, "分环节问题诊断", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 27, "结合逐类别覆盖、候选池污染和聚类指标定位误差来源，为后续针对性改进提供依据。", 14)
    set_shape_text(slide, 35, "可复现实验规范", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 33, "固定类别划分、校准条件和重复实验设置，保证不同方法的比较具有一致性和可复核性。", 14)
    for heading_index in (11, 17, 23, 29, 35):
        set_runs_style(slide, heading_index, RGBColor(255, 255, 255), bold=True)

    # 6. Research problem and route.
    slide = prs.slides[3]
    set_nav_state(slide, 2)
    set_title(slide, 7, "3.1 实验问题与研究路线")
    set_shape_text(
        slide, 9,
        "核心问题：语义相近的新类别可能与已知类别共享相似特征，导致未知检测和候选聚类结果不稳定。\n"
        "• 表征重叠：未知样本与已知类别在特征空间和拒识分数上重叠；\n"
        "• 指标脱节：AUROC 的排序改善不一定转化为固定工作点的拒识收益；\n"
        "• 候选污染：漏检与误检共同影响后续新类聚类。",
        14,
    )
    set_shape_text(slide, 13, "当前问题", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 11, "研究路线", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(
        slide, 12,
        "① 固定类别划分与评价协议，建立 scratch CNN 和 standard-KD 基线；\n"
        "② 对比 shared-backbone，检查拒识表征是否改善；\n"
        "③ 审计 uncertainty-KD 的权重、损失与梯度；\n"
        "④ 结合逐类别覆盖、候选污染和聚类指标定位瓶颈。",
        14,
    )

    # 5. Work completed: describe actual implementation in chronological order.
    slide = prs.slides[4]
    set_nav_state(slide, 2)
    set_title(slide, 39, "3.2 已完成的主要工作")
    remove_shapes(slide, tuple(range(7, 39)) + (40,))
    subtitle = slide.shapes.add_textbox(Inches(0.86), Inches(1.80), Inches(11.65), Inches(0.32))
    set_new_text(
        subtitle,
        "围绕数据协议、模型训练、开放识别和结果验证，已形成可运行的实验链路。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )
    add_work_card(
        slide, 0.82, 2.25, 5.75, 2.05, "01", "统一实验协议",
        "完成 CIFAR-100 的 60/40 类别划分及 semantic-hard 划分；固定数据增强、训练/验证/测试隔离和评价规则，为不同方法的公平对照提供统一条件。",
    )
    add_work_card(
        slide, 6.76, 2.25, 5.75, 2.05, "02", "建立模型对照",
        "完成 scratch CNN、shared-backbone、standard-KD 和 uncertainty-KD 的训练入口；统一输入、训练设置和输出表征，比较不同表征学习方案对未知检测的影响。",
    )
    add_work_card(
        slide, 0.82, 4.62, 5.75, 2.05, "03", "完成识别与聚类链路",
        "实现分类、不确定性估计、NNPU 拒识、候选筛选和 PCA + KMeans 聚类；未知标签不参与训练和阈值校准，仅用于最终指标计算。",
    )
    add_work_card(
        slide, 6.76, 4.62, 5.75, 2.05, "04", "完成对照与训练审计",
        "在三组类别划分上记录分类、AUROC、未知类别拒识率和聚类指标；核对 uncertainty-KD 的权重、损失和梯度是否真正参与优化。",
    )

    # 6. Stage findings.
    slide = prs.slides[5]
    set_nav_state(slide, 2)
    set_title(slide, 0, "3.3 当前阶段性发现")
    set_shape_text(slide, 11, "shared-backbone", 18, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 9, "在三组不同类别划分的重复实验中，shared-backbone 将 AUROC 从 0.7675 提升至 0.7995，平均提高 0.0320；未知类别拒识率从 21.55% 提升至 25.16%，提高 3.61 个百分点。三组实验方向一致。", 12)
    set_shape_text(slide, 23, "uncertainty-KD", 18, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 21, "训练审计表明，uncertainty-KD 的不确定性权重、蒸馏损失和梯度均已进入优化过程。但三组重复实验中未知类别拒识率“两降一升”，尚未形成稳定的任务收益。", 12)
    set_shape_text(slide, 29, "综合判断", 18, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 27, "当前流程已能够完成分类、未知检测、候选筛选和新类聚类。shared-backbone 具有局部拒识收益；uncertainty-KD 暂保留为消融方案。", 12)
    resize_shape(slide, 7, x=0.78, y=1.95, w=3.78, h=4.85)
    resize_shape(slide, 19, x=4.78, y=1.95, w=3.78, h=4.85)
    resize_shape(slide, 25, x=8.78, y=1.95, w=3.78, h=4.85)
    resize_shape(slide, 11, x=1.05, y=2.45, w=3.20, h=0.45)
    resize_shape(slide, 9, x=1.05, y=3.15, w=3.20, h=2.70)
    resize_shape(slide, 23, x=5.05, y=2.45, w=3.20, h=0.45)
    resize_shape(slide, 21, x=5.05, y=3.15, w=3.20, h=2.70)
    resize_shape(slide, 29, x=9.05, y=2.45, w=3.20, h=0.45)
    resize_shape(slide, 27, x=9.05, y=3.15, w=3.20, h=2.70)
    set_shape_text(slide, 9, "三组类别划分的改善方向一致。", 10)
    set_shape_text(slide, 21, "三组实验呈“两降一升”，任务收益尚不稳定。", 10)
    resize_shape(slide, 9, x=1.05, y=5.72, w=3.20, h=0.48)
    resize_shape(slide, 21, x=5.05, y=5.72, w=3.20, h=0.48)
    add_visual_table(
        slide, 1.00, 3.18, 3.30, 2.18,
        [
            ["指标", "对照", "改进后", "变化"],
            ["AUROC", "0.7675", "0.7995", "+0.0320"],
            ["未知拒识率", "21.55%", "25.16%", "+3.61 个百分点"],
        ],
        8,
    )
    add_visual_table(
        slide, 5.00, 3.18, 3.30, 2.18,
        [
            ["划分", "对照", "uncertainty-KD", "结果"],
            ["1", "29.63%", "22.90%", "下降"],
            ["2", "30.55%", "26.80%", "下降"],
            ["3", "28.95%", "29.48%", "上升"],
        ],
        8,
    )
    for heading_index in (11, 23, 29):
        set_runs_style(slide, heading_index, TEMPLATE_NAVY, size=18, bold=True)
    for body_index in (9, 21, 27):
        set_runs_style(slide, body_index, RGBColor(51, 51, 51), size=10)
    remove_shapes(slide, (8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36))

    # 7. Progress and bottleneck diagnosis.
    slide = prs.slides[6]
    set_nav_state(slide, 2)
    set_title(slide, 0, "3.4 实验进展与问题")
    remove_shapes(slide, tuple(range(6, 21)))
    remove_shapes(slide, (6, 7))
    add_work_card(
        slide, 0.82, 2.05, 3.82, 4.45, "01", "结果复核",
        "第一轮多划分对照已经完成。shared-backbone 在三组类别划分中对 AUROC 和未知类别拒识率的改善方向一致；uncertainty-KD 的权重、蒸馏损失和梯度已通过训练审计，但任务收益仍存在波动。",
    )
    add_work_card(
        slide, 4.76, 2.05, 3.82, 4.45, "02", "问题定位",
        "语义相近的新类别仍可能落在已知类别的支持区域内，造成未知样本漏检和候选覆盖不足。漏检与误检还会共同影响后续聚类，因此 AUROC、固定工作点拒识率和聚类结果并不总是同步变化。",
    )
    add_work_card(
        slide, 8.70, 2.05, 3.82, 4.45, "03", "下一步重点",
        "研究将从整体指标比较转向逐类别误差分析：统计不同新类别的漏检情况，检查候选池污染程度，并分析已知类别支持区域的边界，进一步判断表征改进能否转化为稳定的新类发现收益。",
    )

    # 8. Experimental flow and condition control.
    slide = prs.slides[7]
    set_nav_state(slide, 2)
    set_title(slide, 0, "3.5 实验流程")
    set_shape_text(slide, 14, "1. 条件固定", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 17, "固定类别划分、训练/验证/测试隔离，明确未知标签不参与训练、模型选择和阈值校准。", 14)
    set_shape_text(slide, 15, "2. 训练与对照", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 22, "运行 scratch CNN、shared-backbone、standard-KD 和 uncertainty-KD 对照。", 14)
    set_shape_text(slide, 16, "3. 拒识与候选", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 23, "完成拒识、候选筛选和聚类模块，记录 AUROC、未知拒识率、已知效用和聚类指标。", 14)
    set_shape_text(slide, 26, "4. 聚类与推理", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 28, "将分类、拒识、候选筛选和新类聚类串联，形成统一实验闭环。", 14)
    set_shape_text(slide, 31, "5. 结果复核", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 32, "开展多划分重复实验、训练审计与逐类别误差分析。", 14)
    set_shape_text(slide, 33, "6. 记录与总结", 14, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 36, "整理结果、修改报告和程序说明，保留可复现实验配置。", 14)
    for heading_index in (14, 15, 16, 26, 31, 33):
        set_runs_style(slide, heading_index, TEMPLATE_NAVY, bold=True)
    for body_index in (17, 22, 23, 28, 32, 36):
        set_runs_style(slide, body_index, RGBColor(51, 51, 51))

    # 9. Project implementation progress.
    slide = prs.slides[8]
    set_nav_state(slide, 2)
    set_title(slide, 0, "3.6 实施进度与当前节点")
    set_shape_text(slide, 7, "2026.06—2026.07", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 6, "研究准备与方案设计：完成文献调研、明确问题定义和整体技术路线。", 13)
    set_shape_text(slide, 10, "2026.08—2026.09", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 9, "数据与基础环境：完成类别划分、数据预处理和基础训练环境，建立可复现实验入口。", 13)
    set_shape_text(slide, 13, "2026.10—2026.11", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 12, "核心模块初步实现：完成教师—学生训练、拒识分支和第一轮对照实验，当前进入结果复核阶段。", 13)
    set_shape_text(slide, 26, "2027.02—2027.03", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 25, "统一框架完善：补充多划分和消融实验，统一记录分类、拒识和聚类指标。", 13)
    set_shape_text(slide, 22, "2026.12—2027.01", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 21, "结果复核与误差分析：对不同模块的对照结果进行复核，开展逐类别覆盖、候选污染和支持区域边界分析。", 13)
    set_shape_text(slide, 24, "2027.04—2027.05", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 23, "综合实验与报告整理：完成综合实验、图表整理、研究报告和结题材料初稿。", 13)
    set_shape_text(slide, 28, "2027.05—2027.06", 13, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 27, "成果汇总与验收准备：完成答辩展示、结题材料定稿和项目验收准备。", 13)
    for date_index in (7, 10, 13, 22, 26, 24, 28):
        set_runs_style(slide, date_index, TEMPLATE_NAVY, bold=True)
    for body_index in (6, 9, 12, 21, 25, 23, 27):
        set_runs_style(slide, body_index, RGBColor(51, 51, 51))

    # 10. Results.
    slide = prs.slides[9]
    set_nav_state(slide, 3)
    set_title(slide, 6, "4.1 阶段性判断")
    set_shape_text(slide, 22, "已形成实验闭环", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 25, "已完成从类别划分、模型训练、未知检测、候选筛选到新类聚类的流程，并固定训练、校准和评价条件。", 14)
    set_shape_text(slide, 23, "已获得局部证据", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 26, "shared-backbone 在三组类别划分中 AUROC 和未知类别拒识率均改善，支持继续研究共享表征。", 14)
    set_shape_text(slide, 24, "尚未形成稳定收益", 16, TEMPLATE_NAVY, True, PP_ALIGN.CENTER)
    set_shape_text(slide, 27, "uncertainty-KD 已真实执行，但三组重复实验两降一升；语义相近新类别仍是当前主要瓶颈。", 14)
    for heading_index in (22, 23, 24):
        set_runs_style(slide, heading_index, TEMPLATE_NAVY, bold=True)
    for body_index in (25, 26, 27):
        set_runs_style(slide, body_index, RGBColor(51, 51, 51))

    # 11. Keep the original team slide content and template, but use the requested wording.
    slide = prs.slides[10]
    set_nav_state(slide, 3)
    set_title(slide, 0, "4.2 团队介绍与分工")
    set_shape_text(slide, 9, "• 项目负责人：王笑颜\n• 项目成员：李柯颖、齐誉涵", 14)
    set_shape_text(slide, 13, "王笑颜：负责项目整体方案设计、进度统筹、开放集识别与知识蒸馏核心算法实现、框架集成与调试。\n"
                          "李柯颖：负责数据集划分、数据增强、模型训练、实验运行、指标计算与结果可视化。\n"
                          "齐誉涵：负责文献整理、代码调试、实验记录、项目材料汇总、研究报告与结题文档撰写。", 14)
    set_runs_style(slide, 9, RGBColor(51, 51, 51))
    set_runs_style(slide, 13, RGBColor(51, 51, 51))

    # 12. Conclusion.
    slide = prs.slides[11]
    set_title(slide, 0, "5.1 中期结论")
    set_shape_text(
        slide, 8,
        "项目推进：已完成研究协议、数据处理、基线模型、教师—学生蒸馏、未知检测、候选筛选和新类聚类，形成了可运行的完整实验链路。\n"
        "方法验证：shared-backbone 在三组类别划分中均改善 AUROC 和未知类别拒识率；训练审计确认 uncertainty-KD 的权重、蒸馏损失和梯度真实进入优化。\n"
        "阶段判断：uncertainty-KD 在三组重复实验中两降一升，尚未形成稳定的任务层面收益，因此暂保留为消融方案。\n"
        "后续重点：围绕语义相近新类别的表征重叠、逐类别候选覆盖、支持区域边界和聚类稳定性继续验证。",
        16,
    )
    set_runs_style(slide, 8, RGBColor(51, 51, 51))

    # Insert the restored section 2 after the introduction section.
    move_slide_to(prs, section_21, 3)
    move_slide_to(prs, section_22, 4)

    # 15. Keep the original closing slide unchanged.
    prs.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期汇报（开题模板复用版）"
    prs.core_properties.subject = "中期进展汇报"
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
