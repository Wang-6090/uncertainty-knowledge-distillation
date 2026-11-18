from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(ROOT.glob("*中期汇报.pptx"))
OUTPUT = ROOT / "最新版_基于不确定性知识蒸馏的新类发现方法中期汇报_算法顺序完整版.pptx"

NAVY = RGBColor(50, 74, 122)
INK = RGBColor(36, 45, 58)
BODY = RGBColor(51, 51, 51)
LIGHT = RGBColor(247, 249, 253)
LINE = RGBColor(196, 209, 230)
ORANGE = RGBColor(239, 119, 42)
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(95, 108, 124)


def add_text(slide, x, y, w, h, text, size=14, color=BODY, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(2)
        run = p.add_run()
        run.text = line
        run.font.name = "微软雅黑"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.bold = bold
    return box


def add_box(slide, x, y, w, h, title, body, fill=LIGHT, title_fill=NAVY,
            title_size=15, body_size=11.5):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = LINE
    box.line.width = Pt(1.1)
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.42))
    header.fill.solid()
    header.fill.fore_color.rgb = title_fill
    header.line.fill.background()
    add_text(slide, x + 0.12, y + 0.04, w - 0.24, 0.32, title, title_size, WHITE, True,
             PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_text(slide, x + 0.12, y + 0.52, w - 0.24, h - 0.62, body, body_size, BODY, False,
             PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    return box


def add_arrow(slide, x, y, w, h, text=""):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = ORANGE
    arrow.line.fill.background()
    if text:
        add_text(slide, x - 0.05, y - 0.32, w + 0.1, 0.25, text, 9, MUTED, False, PP_ALIGN.CENTER)
    return arrow


def add_nav(slide, active=0):
    labels = ["项目绪论", "研创核心", "方案实施", "成果保障", "总结"]
    xs = [4.36, 6.15, 7.95, 9.74, 11.53]
    for i, (label, x) in enumerate(zip(labels, xs)):
        if i == active:
            rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(0.02), Inches(1.78), Inches(0.48))
            rect.fill.solid()
            rect.fill.fore_color.rgb = NAVY
            rect.line.fill.background()
        add_text(slide, x, 0.08, 1.78, 0.32, label, 20, WHITE if i == active else RGBColor(191, 191, 191), True,
                 PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.61), Inches(13.333), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()


def replace_text(shape, text, size=14, color=BODY, bold=False, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = "微软雅黑"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.bold = bold


def copy_slide(prs, source_slide):
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    for shape in source_slide.shapes:
        slide.shapes._spTree.insert_element_before(deepcopy(shape._element), "p:extLst")
    return slide


def move_slide(prs, slide, pos):
    ids = prs.slides._sldIdLst
    element = next(item for item in ids if int(item.id) == slide.slide_id)
    ids.remove(element)
    ids.insert(pos, element)


def remove_slide(prs, slide):
    ids = prs.slides._sldIdLst
    element = next(item for item in ids if int(item.id) == slide.slide_id)
    ids.remove(element)


def set_existing_nav_active(slide, active=0):
    """Update the active tab on a slide copied from the original template."""
    labels = ["项目绪论", "研创核心", "方案实施", "成果保障", "总结"]
    xs = [4.36, 6.15, 7.95, 9.74, 11.53]
    for shape in slide.shapes:
        if not hasattr(shape, "text") or not shape.text.strip():
            continue
        label = shape.text.strip()
        if label not in labels:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = WHITE if labels.index(label) == active else RGBColor(191, 191, 191)
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            continue
        if shape.top < Inches(0.15) and Inches(1.55) < shape.width < Inches(2.10) and Inches(0.45) < shape.height < Inches(0.75):
            shape.left = Inches(xs[active])
            shape.fill.solid()
            shape.fill.fore_color.rgb = NAVY
            shape.line.fill.background()
            break


def add_model_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=1)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "2.1 模型结构与输入输出", 24, INK, True)
    add_text(slide, 0.78, 1.38, 11.8, 0.30,
             "先由教师模型指导学生模型学习，再由学生模型输出分类表征、拒识表征和不确定性信号。",
             12.5, MUTED, False, PP_ALIGN.CENTER)

    add_text(slide, 0.82, 1.82, 11.70, 0.28, "训练阶段：教师模型指导学生模型学习", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_box(slide, 0.82, 2.12, 2.35, 1.12, "输入图像", "CIFAR-100\n已知类标注 / 未知类无标注", title_fill=NAVY, title_size=14, body_size=10.5)
    add_arrow(slide, 3.40, 2.50, 0.50, 0.30)
    add_box(slide, 4.05, 2.12, 2.45, 1.12, "教师模型", "ResNet-34\n输出软标签与分类语义", title_fill=NAVY, title_size=14, body_size=10.5)
    add_arrow(slide, 6.75, 2.50, 0.50, 0.30, "知识蒸馏")
    add_box(slide, 7.40, 2.12, 2.45, 1.12, "学生模型", "ResNet-18\n学习可迁移特征表示", title_fill=NAVY, title_size=14, body_size=10.5)
    add_text(slide, 10.20, 2.24, 2.28, 0.84, "教师：提供知识\n学生：学习表征", 12, MUTED, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

    add_text(slide, 0.82, 3.68, 11.70, 0.28, "模型输出：分类路径与拒识路径并行产生结果", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_box(slide, 0.82, 4.05, 2.25, 1.35, "同一输入图像", "分别进入分类路径\n和拒识路径", title_fill=NAVY, title_size=14, body_size=10.5)
    add_arrow(slide, 3.20, 4.50, 0.28, 0.30)

    output = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.55), Inches(3.98), Inches(7.30), Inches(1.50))
    output.fill.solid()
    output.fill.fore_color.rgb = LIGHT
    output.line.color.rgb = LINE
    output.line.width = Pt(1.1)
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.55), Inches(3.98), Inches(7.30), Inches(0.42))
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()
    add_text(slide, 3.70, 4.02, 7.00, 0.30, "模型的两条输出路径", 14, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_text(slide, 3.82, 4.48, 6.76, 0.84, "分类路径：ResNet-18 → 已知类别 logits 与分类特征\n拒识路径：拒识编码器 → 拒识特征与拒识分数\n不确定性信号：分类置信度、预测熵或不确定性头分数",
             11.5, BODY, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

    add_arrow(slide, 11.05, 4.50, 0.28, 0.30, "后续进入拒识与聚类")
    add_box(slide, 11.48, 4.05, 1.12, 1.35, "后续处理", "拒识\n→聚类", title_fill=ORANGE, title_size=11.5, body_size=10)

    add_text(slide, 0.84, 5.88, 11.70, 0.42,
             "本页只说明模型输出；拒识评分、候选筛选、PCA 和 KMeans 的具体算法顺序见后续页面。",
             11.5, MUTED, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    return slide


def add_method_overview_slide(prs):
    """Create a single, left-to-right overview of the research method."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=0)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "1.2 总体研究方法", 24, INK, True)
    add_text(
        slide, 0.80, 1.38, 11.75, 0.34,
        "围绕“已知类别学习—未知检测—新类形成”建立统一的开放环境实验流程。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )

    steps = [
        (0.76, "开放环境协议", "已知类别用于训练；\n未知类别只在测试阶段评价。", NAVY),
        (3.88, "教师—学生学习", "教师提供软标签；\n学生学习开放环境表征。", NAVY),
        (7.00, "未知检测", "不确定性估计与拒识分支\n筛选未知候选。", ORANGE),
        (10.12, "新类形成", "对候选特征做 PCA 降维，\n再用 KMeans 聚类。", ORANGE),
    ]
    for x, title, body, color in steps:
        add_box(slide, x, 2.38, 2.46, 1.60, title, body, fill=LIGHT, title_fill=color, title_size=14, body_size=11)
    for x in [3.36, 6.48, 9.60]:
        add_arrow(slide, x, 2.93, 0.42, 0.28)

    add_text(slide, 0.84, 4.62, 11.70, 0.30, "方法之间的关系", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_text(
        slide, 1.10, 5.06, 11.10, 0.72,
        "分类表征是后续未知检测的基础；不确定性和拒识分支负责筛选候选；候选筛选结果再进入降维与聚类，形成新类别。",
        16, BODY, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide, 1.25, 6.25, 10.85, 0.34,
        "后续 2.1 介绍模型结构，2.2 说明训练对照，2.3 说明拒识，2.4 说明候选聚类，2.5 总结重点探索。",
        11.5, MUTED, False, PP_ALIGN.CENTER,
    )
    return slide


def add_training_objective_slide(prs):
    """Explain the controlled model comparisons rather than a fake serial flow."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=1)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "2.2 模型对照与改进关系", 24, INK, True)
    add_text(
        slide, 0.80, 1.38, 11.75, 0.34,
        "上方比较分类与蒸馏方案；下方单独比较拒识编码器的初始化方式。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )

    cards = [
        (0.76, "Scratch CNN", "不使用教师模型\n只用已知类别标签训练\n提供基础参照", NAVY),
        (4.48, "standard-KD", "在基础模型上加入教师模型\n使用温度缩放 KL 散度\n传递软类别分布", NAVY),
        (8.20, "uncertainty-KD", "在 standard-KD 上改进\n按教师不确定性加权 KL\n检验迁移是否更可靠", ORANGE),
    ]
    for x, title, body, color in cards:
        add_box(slide, x, 2.08, 3.42, 1.82, title, body, fill=LIGHT, title_fill=color, title_size=14.5, body_size=11.5)
    for x in [3.98, 7.70]:
        add_arrow(slide, x, 2.75, 0.38, 0.26)
    add_text(slide, 3.45, 2.35, 0.98, 0.22, "加入教师知识", 9.5, MUTED, False, PP_ALIGN.CENTER)
    add_text(slide, 7.17, 2.35, 0.98, 0.22, "加入不确定性加权", 9.5, MUTED, False, PP_ALIGN.CENTER)

    add_text(slide, 0.82, 4.38, 11.70, 0.30, "训练阶段的输出", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_box(slide, 1.04, 4.82, 4.72, 1.28, "分类表征", "分类损失与知识蒸馏共同塑造\n为后续拒识提供基础特征", title_fill=NAVY, title_size=13, body_size=10.5)
    add_box(slide, 7.52, 4.82, 4.72, 1.28, "不确定性信号", "教师不确定性用于 KD 加权\n分类熵和不确定性头分数供后续评分", title_fill=ORANGE, title_size=13, body_size=10.5)
    add_text(slide, 5.90, 5.22, 1.52, 0.42, "输出给\n下一环节", 11, MUTED, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_text(slide, 1.25, 6.32, 10.85, 0.30, "完成特征学习后，模型进入拒识编码器、NNPU 风险估计和候选筛选流程。", 11.5, MUTED, False, PP_ALIGN.CENTER)
    return slide


def add_rejection_algorithm_slide(prs):
    """Explain the rejection and clustering algorithm as an explicit data flow."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=1)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "2.3 未知拒识与候选筛选", 24, INK, True)
    add_text(
        slide, 0.80, 1.38, 11.75, 0.34,
        "先判断样本是否偏离已知类别支持区域，再把通过拒识的样本交给后续聚类。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )

    steps = [
        (0.66, "① 提取拒识特征", "学生模型输出\n拒识表征与不确定性", NAVY),
        (3.13, "② 初始化拒识器", "scratch_cnn：随机初始化\nshared-backbone：主干初始化", NAVY),
        (5.60, "③ NNPU 风险", "已知样本有标签\n混合发现池无标签", ORANGE),
        (8.07, "④ 校准工作点", "只用已知验证集\n确定拒识阈值", ORANGE),
        (10.54, "⑤ 筛选候选", "超过拒识工作点的\n样本进入候选池", ORANGE),
    ]
    for x, title, body, color in steps:
        add_box(slide, x, 2.26, 2.10, 1.62, title, body, fill=LIGHT, title_fill=color, title_size=12.5, body_size=10.2)
    for x in [2.84, 5.31, 7.78, 10.25]:
        add_arrow(slide, x, 2.86, 0.26, 0.22)

    add_text(slide, 0.84, 4.48, 11.70, 0.30, "NNPU 在拒识中的作用", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_text(
        slide, 1.05, 4.92, 11.25, 0.72,
        "已知样本提供可靠的已知侧信息；混合发现池同时含有已知和未知样本，因此只作为未标注混合分布。NNPU 在风险估计中扣除混合池中可能来自已知分布的部分，并对负的未知风险进行非负截断，训练拒识器输出未知分数。shared-backbone 只改变拒识编码器的初始化方式，初始化后仍独立训练。",
        13, BODY, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide, 1.12, 6.16, 11.10, 0.34,
        "未知标签不参与拒识器训练、阈值校准和候选规则选择，只在最终评价阶段使用。",
        11.5, MUTED, False, PP_ALIGN.CENTER,
    )
    return slide


def add_candidate_clustering_slide(prs):
    """Explain the post-rejection candidate clustering stage."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=1)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "2.4 候选聚类与全链路评价", 24, INK, True)
    add_text(
        slide, 0.80, 1.38, 11.75, 0.34,
        "候选池确定后，才进入特征整理、聚类和新类评价；聚类不能弥补前面的未知漏检。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )
    steps = [
        (0.90, "候选池", "拒识通过的\n未知候选", NAVY),
        (3.35, "特征提取", "使用候选样本的\n拒识/投影特征", NAVY),
        (5.80, "PCA 降维", "压缩冗余维度\n保留主要变化方向", ORANGE),
        (8.25, "KMeans 聚类", "按特征相似性\n形成候选类别簇", ORANGE),
        (10.70, "指标评价", "ACC、NMI、ARI\n结合覆盖与污染分析", ORANGE),
    ]
    for x, title, body, color in steps:
        add_box(slide, x, 2.32, 1.86, 1.55, title, body, fill=LIGHT, title_fill=color, title_size=12.5, body_size=10.2)
    for x in [2.98, 5.43, 7.88, 10.33]:
        add_arrow(slide, x, 2.90, 0.25, 0.22)

    add_text(slide, 0.84, 4.45, 11.70, 0.30, "为什么要分开评价候选池和聚类", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_box(slide, 1.00, 4.86, 5.20, 1.34, "候选池质量", "未知覆盖率：真正的新类别有多少进入候选池\n已知污染率：候选池中混入多少已知样本", title_fill=NAVY, title_size=13, body_size=10.5)
    add_box(slide, 7.12, 4.86, 5.20, 1.34, "聚类质量", "在候选池固定后，评价类别簇是否紧凑、分离\n使用 ACC、NMI、ARI 等指标进行回顾性分析", title_fill=ORANGE, title_size=13, body_size=10.5)
    add_text(slide, 1.20, 6.38, 10.95, 0.28, "未知标签只用于最终评价，不用于候选筛选、类别数选择或阈值校准。", 11.5, MUTED, False, PP_ALIGN.CENTER)
    return slide


def add_innovation_slide(prs):
    """Create a focused innovation slide with three research lines."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_nav(slide, active=1)
    add_text(slide, 0.62, 0.82, 12.0, 0.52, "2.5 核心创新与研究重点", 24, INK, True)
    add_text(
        slide, 0.80, 1.38, 11.75, 0.34,
        "在模型和算法流程明确之后，进一步验证不确定性蒸馏、共享表征拒识和全链路评价。",
        13, MUTED, False, PP_ALIGN.CENTER,
    )

    cards = [
        (
            0.72,
            "不确定性驱动知识蒸馏",
            "以样本预测不确定性调节蒸馏强度，使教师知识迁移与样本可靠性相匹配。\n\n重点验证：不确定性权重、蒸馏损失和梯度是否真正参与训练，以及收益是否稳定。",
            ORANGE,
        ),
        (
            4.56,
            "共享表征支持未知拒识",
        "以学生分类主干初始化拒识编码器，再独立训练拒识表征，检验分类知识能否迁移到未知检测。\n\n重点比较：scratch_cnn 与 shared-backbone 初始化的拒识表现。",
            NAVY,
        ),
        (
            8.40,
            "全链路诊断与评价",
            "将分类、未知检测、候选筛选和新类聚类放入同一评价流程，避免只看单一指标。\n\n重点记录：AUROC、未知拒识率、候选覆盖、候选污染和聚类结果。",
            NAVY,
        ),
    ]
    for x, title, body, color in cards:
        add_box(slide, x, 2.08, 3.48, 3.82, title, body, fill=LIGHT, title_fill=color, title_size=14, body_size=11)

    add_text(slide, 0.82, 6.26, 1.35, 0.30, "核心判断", 12.5, NAVY, True, PP_ALIGN.CENTER)
    add_text(
        slide, 2.30, 6.20, 10.05, 0.44,
        "shared-backbone 重点检验开放表征的拒识作用；uncertainty-KD 重点检验机制是否能转化为稳定的任务收益。",
        11.5, BODY, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE,
    )
    return slide


def main():
    prs = Presentation(SOURCE)
    # Compress the front matter so the model structure and experimental findings
    # carry the presentation rather than repeated background prose.
    background = prs.slides[1]
    replace_text(
        background.shapes[10],
        "封闭集模型假设训练与测试类别一致；面对训练阶段未出现的新类别，容易将未知样本误判为已知类别。项目关注未知检测与新类发现的连续任务。",
        14,
    )
    replace_text(background.shapes[6], "1.1 研究背景与问题定义", 24, INK, True)
    replace_text(background.shapes[15], "问题定义", 14, NAVY, True)
    replace_text(background.shapes[18], "研究目标", 14, NAVY, True)
    replace_text(background.shapes[19], "核心任务：", 14, NAVY, True)
    replace_text(background.shapes[20], "关键难点：", 14, NAVY, True)
    replace_text(background.shapes[21], "未知类别与已知类别可能存在特征重叠；不确定性如何辅助未知检测并支持后续聚类，是本项目要解决的核心问题。", 12.5)
    replace_text(background.shapes[22], "建立包含已知分类、未知检测、候选筛选和新类聚类的统一实验框架，分析不确定性估计与知识蒸馏的作用。", 12.5)

    # Use the formerly empty middle area to summarize the task in one compact
    # visual chain.  This keeps the introduction concise while making the
    # transition from background to project objective explicit.
    add_text(background, 0.82, 3.56, 11.72, 0.27, "项目任务链路", 12, NAVY, True, PP_ALIGN.CENTER)
    add_box(background, 1.28, 3.92, 2.62, 0.64, "已知类别学习", "学习类别判别表征", fill=LIGHT, title_fill=NAVY,
            title_size=11.5, body_size=9.5)
    add_arrow(background, 4.08, 4.10, 0.48, 0.24)
    add_box(background, 4.72, 3.92, 2.62, 0.64, "未知检测", "判断样本是否偏离已知支持", fill=LIGHT, title_fill=NAVY,
            title_size=11.5, body_size=9.5)
    add_arrow(background, 7.52, 4.10, 0.48, 0.24)
    add_box(background, 8.16, 3.92, 2.62, 0.64, "新类发现", "筛选候选并形成类别簇", fill=LIGHT, title_fill=ORANGE,
            title_size=11.5, body_size=9.5)

    old_method = prs.slides[2]
    old_research = prs.slides[3]
    focus_slide = prs.slides[4]
    findings_slide = prs.slides[7]
    conclusion_slide = prs.slides[11]
    judgment_slide = prs.slides[9]
    team_slide = prs.slides[10]
    replace_text(
        findings_slide.shapes[9],
        "在 CIFAR-100 semantic-hard 60/40 协议下，三套不同的已知—未知类别划分中，shared-backbone 的 AUROC 从 0.7675 提升至 0.7995（+0.0320），未知类别拒识率从 21.55% 提升至 25.16%（+3.61 个百分点）。",
        11.5,
    )
    replace_text(
        findings_slide.shapes[12],
        "uncertainty-KD 已进入训练，但三套划分的拒识率呈“两降一升”，尚未形成稳定的任务收益，暂作为消融方案。",
        12,
    )
    replace_text(
        findings_slide.shapes[15],
        "当前已完成分类—未知检测—候选聚类闭环；shared-backbone 在未知检测环节出现一致改善，uncertainty-KD 仍需结合候选覆盖和聚类结果继续复核。",
        12,
    )
    replace_text(
        conclusion_slide.shapes[8],
        "中期结论：项目已建立可运行、可复核的分类—未知检测—候选聚类流程；shared-backbone 获得了较稳定的局部拒识证据；当前主要瓶颈是语义相近类别的表征重叠和候选池质量。下一阶段将围绕类别边界、候选覆盖和聚类稳定性继续验证。",
        14,
    )
    replace_text(judgment_slide.shapes[6], "4.1 后续研究重点与风险控制", 23, INK, True)
    replace_text(judgment_slide.shapes[22], "语义相近类别", 13.5, NAVY, True)
    replace_text(judgment_slide.shapes[23], "候选池质量", 13.5, NAVY, True)
    replace_text(judgment_slide.shapes[24], "蒸馏收益稳定性", 13.5, NAVY, True)
    replace_text(judgment_slide.shapes[25], "分析新类别与已知类别的特征和拒识分数重叠，定位持续漏检的类别。", 12.5)
    replace_text(judgment_slide.shapes[26], "分别记录未知覆盖率和已知污染率，分析候选池质量对后续聚类的影响。", 12.5)
    replace_text(judgment_slide.shapes[27], "复核不确定性来源、权重分布和跨划分结果，决定是否纳入默认方案。", 12.5)
    method_slide = add_method_overview_slide(prs)
    model_slide = add_model_slide(prs)
    training_slide = add_training_objective_slide(prs)
    rejection_slide = add_rejection_algorithm_slide(prs)
    candidate_slide = add_candidate_clustering_slide(prs)
    innovation_slide = add_innovation_slide(prs)
    remove_slide(prs, old_method)
    remove_slide(prs, old_research)
    remove_slide(prs, focus_slide)
    remove_slide(prs, team_slide)
    move_slide(prs, method_slide, 2)
    move_slide(prs, model_slide, 3)
    move_slide(prs, training_slide, 4)
    move_slide(prs, rejection_slide, 5)
    move_slide(prs, candidate_slide, 6)
    move_slide(prs, innovation_slide, 7)
    prs.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期汇报（模型结构强化版）"
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
