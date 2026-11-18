from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "midterm_visual_assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#324A7A"
BLUE = "#5B84C4"
LIGHT_BLUE = "#DDE8F6"
ORANGE = "#D58B4D"
LIGHT_ORANGE = "#F5E2CF"
GREEN = "#4C9A78"
LIGHT_GREEN = "#DDEFE7"
GRAY = "#5F6C7C"
LIGHT_GRAY = "#F4F6F9"

FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-Regular.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    CJK_FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    CJK_FONT = "DejaVu Sans"


def save_csv(name: str, rows: list[list[str]]) -> Path:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    return path


def setup_plot(title: str):
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=22, color=NAVY, fontweight="bold", pad=20)
    return fig, ax


def make_shared_chart() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.0), dpi=180)
    fig.patch.set_facecolor("white")
    values = [(0.7675, 0.7995), (21.55, 25.16)]
    titles = ["AUROC", "Unknown rejection rate (%)"]
    labels = ["Baseline", "shared-backbone"]
    for ax, pair, title in zip(axes, values, titles):
        bars = ax.bar(labels, pair, color=["#B7C0CC", BLUE], width=0.55)
        ax.set_title(title, color=NAVY, fontsize=17, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, pair):
            suffix = "%" if title.endswith("(%)") else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.2f}{suffix}", ha="center", va="bottom",
                    fontsize=13, color="#333333", fontweight="bold")
        if title == "AUROC":
            ax.set_ylim(0.70, 0.84)
        else:
            ax.set_ylim(0, 30)
    fig.suptitle("shared-backbone 相较对照方案的指标变化", fontsize=23,
                 color=NAVY, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "AUROC：0.7675 → 0.7995（+0.0320）    未知类别拒识率：21.55% → 25.16%（+3.61 个百分点）",
             ha="center", fontsize=13, color=GRAY)
    path = OUT / "shared_backbone_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_uncertainty_chart() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(12.8, 6.6), dpi=180)
    fig.patch.set_facecolor("white")
    splits = ["Split 1", "Split 2", "Split 3"]
    baseline = [29.63, 30.55, 28.95]
    kd = [22.90, 26.80, 29.48]
    x = range(len(splits))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], baseline, width, label="Baseline", color="#B7C0CC")
    b2 = ax.bar([i + width / 2 for i in x], kd, width, label="uncertainty-KD", color=ORANGE)
    ax.set_xticks(list(x), splits)
    ax.set_ylabel("Unknown rejection rate (%)", fontsize=13)
    ax.set_ylim(0, 36)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("uncertainty-KD 三组类别划分对比", color=NAVY, fontsize=22, fontweight="bold", pad=18)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=12)
    ax.text(0.02, 0.96, "结论：两组下降、一组上升，任务收益尚不稳定",
            transform=ax.transAxes, va="top", fontsize=14, color=ORANGE, fontweight="bold")
    path = OUT / "uncertainty_kd_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def box(ax, x, y, w, h, title, body, face, edge):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                           linewidth=2, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=17, color=NAVY, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=12, color="#333333", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=2, color=color))


def make_stacked_flow(title, items, caption, filename) -> Path:
    """Render a spacious two-column, three-row flow instead of a cramped line."""
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 7.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.93, title, ha="center", fontsize=23,
            color=NAVY, fontweight="bold")

    # Sequence: 1 -> 2 -> 3 -> 4 -> 5, arranged in a readable zig-zag.
    positions = [(0.10, 0.64), (0.58, 0.64), (0.10, 0.38),
                 (0.58, 0.38), (0.34, 0.12)]
    w, h = 0.30, 0.17
    for (x, y), item in zip(positions, items):
        title_text, body_text, face, edge = item
        box(ax, x, y, w, h, title_text, body_text, face, edge)

    arrow(ax, 0.40, 0.725, 0.57, 0.725)
    arrow(ax, 0.73, 0.64, 0.41, 0.47)
    arrow(ax, 0.40, 0.465, 0.57, 0.465)
    arrow(ax, 0.73, 0.38, 0.65, 0.28)
    ax.text(0.5, 0.02, caption, ha="center", fontsize=14, color=GRAY)
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_pipeline() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 5.6), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.93, "开放环境下的新类发现实验流程", ha="center", fontsize=23,
            color=NAVY, fontweight="bold")

    # A single left-to-right pipeline, with short labels so the figure remains
    # readable when inserted into a 16:9 presentation slide.
    items = [
        ("已知训练", "仅用已知标签", LIGHT_BLUE, BLUE),
        ("表征学习", "scratch / KD / shared", LIGHT_BLUE, BLUE),
        ("不确定性", "量化预测不确定性", LIGHT_ORANGE, ORANGE),
        ("未知拒识", "筛选候选样本", LIGHT_ORANGE, ORANGE),
        ("新类聚类", "PCA + KMeans", LIGHT_GREEN, GREEN),
    ]
    xs = [0.025, 0.225, 0.425, 0.625, 0.825]
    for x, item in zip(xs, items):
        box(ax, x, 0.37, 0.15, 0.28, *item)
    for x in [0.175, 0.375, 0.575, 0.775]:
        arrow(ax, x, 0.51, x + 0.045, 0.51)
    ax.text(0.5, 0.16, "训练阶段不使用未知标签；未知标签仅用于最终评价。",
            ha="center", fontsize=14, color=GRAY)
    path = OUT / "open_environment_pipeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_issue_chain() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 7.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.93, "当前核心问题：表征重叠如何影响新类发现", ha="center",
            fontsize=22, color=NAVY, fontweight="bold")

    columns = [
        (0.06, "原因", "语义相近\n表征重叠", LIGHT_ORANGE, ORANGE),
        (0.37, "直接影响", "未知漏检\n候选覆盖不足", LIGHT_GRAY, GRAY),
        (0.68, "后续后果", "候选污染\n聚类不稳定", LIGHT_GREEN, GREEN),
    ]
    for x, header, body, face, edge in columns:
        ax.text(x + 0.13, 0.76, header, ha="center", va="center",
                fontsize=16, color=edge, fontweight="bold")
        box(ax, x, 0.43, 0.26, 0.20, body.split("\n")[0], body.split("\n")[1], face, edge)
    arrow(ax, 0.33, 0.53, 0.36, 0.53, GRAY)
    arrow(ax, 0.64, 0.53, 0.67, 0.53, GRAY)
    ax.text(0.5, 0.20, "评价层面的表现：AUROC、固定工作点拒识率与聚类结果不一定同步变化。",
            ha="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.08, "后续：逐类别统计漏检、检查候选池污染，并分析已知支持区域边界。",
            ha="center", fontsize=14, color=GRAY)
    path = OUT / "current_bottleneck_chain.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_evaluation_framework() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 6.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.92, "三项任务与评价指标的对应关系", ha="center",
            fontsize=23, color=NAVY, fontweight="bold")
    columns = [
        (0.06, "已知类别分类", "判断模型是否保留\n基本分类能力", "分类准确率\n已知类别效用", LIGHT_BLUE, BLUE),
        (0.37, "未知样本拒识", "判断未知样本能否\n被筛出", "AUROC\n未知类别拒识率", LIGHT_ORANGE, ORANGE),
        (0.68, "新类别聚类", "判断候选样本能否\n形成稳定类别", "聚类准确率\n候选覆盖与污染", LIGHT_GREEN, GREEN),
    ]
    for x, title, task, metrics, face, edge in columns:
        box(ax, x, 0.45, 0.26, 0.23, title, task, face, edge)
        ax.text(x + 0.13, 0.27, metrics, ha="center", va="center",
                fontsize=13, color="#333333")
    arrow(ax, 0.33, 0.565, 0.36, 0.565, GRAY)
    arrow(ax, 0.64, 0.565, 0.67, 0.565, GRAY)
    ax.text(0.5, 0.08, "完整任务评价不能只看单一指标，需要同时考察分类、拒识和聚类结果。",
            ha="center", fontsize=14, color=NAVY, fontweight="bold")
    path = OUT / "evaluation_framework.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_method_matrix() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 6.4), dpi=180)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_title("模型与方法对照矩阵", fontsize=23, color=NAVY, fontweight="bold", pad=20)
    rows = [
        ["方法", "主要作用", "当前阶段结论"],
        ["scratch CNN", "建立不蒸馏的基础分类表征", "作为基础对照"],
        ["standard-KD", "迁移教师模型的分类语义", "作为蒸馏对照"],
        ["shared-backbone", "比较共享表征对未知检测的作用", "AUROC 与未知拒识率均改善"],
        ["uncertainty-KD", "按不确定性调整知识迁移", "机制已生效，但任务收益不稳定"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                     colLoc="center", bbox=[0.05, 0.16, 0.90, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C4D1E6")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F7F9FD" if row % 2 else "#EAF0F8")
            cell.get_text().set_color("#333333")
    ax.text(0.5, 0.06, "表中结论仅反映当前阶段实验，不代表完整新类发现问题已经解决。",
            ha="center", fontsize=13, color=GRAY)
    path = OUT / "method_comparison_matrix.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_timeline() -> Path:
    plt.rcParams["font.family"] = CJK_FONT
    plt.rcParams["font.sans-serif"] = [CJK_FONT]
    fig, ax = plt.subplots(figsize=(13.33, 5.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.92, "项目实施进度与当前节点", ha="center", fontsize=23,
            color=NAVY, fontweight="bold")
    ax.plot([0.09, 0.91], [0.49, 0.49], color="#B7C0CC", linewidth=4, solid_capstyle="round")
    phases = [
        (0.10, "2026.06—07", "方案设计", "文献调研\n问题定义"),
        (0.27, "2026.08—09", "数据与环境", "类别划分\n预处理与基础训练"),
        (0.45, "2026.10—11", "核心模块", "模型训练、拒识\n第一轮对照"),
        (0.64, "2026.12—2027.01", "结果复核", "逐类别覆盖\n候选污染分析"),
        (0.82, "2027.02—06", "综合实验与验收", "多划分、消融\n报告与结题"),
    ]
    for i, (x, date, title, body) in enumerate(phases):
        active = i == 2
        color = NAVY if active else GRAY
        ax.scatter([x], [0.49], s=340 if active else 220,
                   color=BLUE if active else "white", edgecolors=color, linewidths=3, zorder=3)
        ax.text(x, 0.65 if i % 2 == 0 else 0.33, date, ha="center", fontsize=12,
                color=color, fontweight="bold")
        ax.text(x, 0.58 if i % 2 == 0 else 0.26, title, ha="center", fontsize=14,
                color=NAVY, fontweight="bold")
        ax.text(x, 0.76 if i % 2 == 0 else 0.12, body, ha="center", fontsize=11,
                color="#333333")
    ax.text(0.45, 0.04, "当前节点：第一轮对照实验完成，进入结果复核阶段。",
            ha="center", fontsize=14, color=NAVY, fontweight="bold")
    path = OUT / "implementation_timeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_text(slide, x, y, w, h, text, size=18, color=NAVY, bold=True):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = 1
    run = p.add_run()
    run.text = text
    run.font.name = "微软雅黑"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))


def add_image_slide(prs, title, image_path):
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_text(slide, 0.5, 0.25, 12.3, 0.45, title, 22, NAVY, True)
    slide.shapes.add_picture(str(image_path), Inches(0.55), Inches(0.90), width=Inches(12.25))
    return slide


def add_table_slide(prs, title, rows):
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    add_text(slide, 0.5, 0.25, 12.3, 0.45, title, 22, NAVY, True)
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(1.0), Inches(1.6), Inches(11.3), Inches(3.8))
    table = table_shape.table
    for c in range(len(rows[0])):
        table.columns[c].width = Inches(11.3 / len(rows[0]))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(50, 74, 122) if r == 0 else RGBColor(247, 249, 253)
            for p in cell.text_frame.paragraphs:
                p.alignment = 1
                for run in p.runs:
                    run.font.name = "微软雅黑"
                    run.font.size = Pt(16 if r == 0 else 15)
                    run.font.bold = r == 0
                    run.font.color.rgb = RGBColor(255, 255, 255) if r == 0 else RGBColor(51, 51, 51)
    return slide


def main():
    shared_csv = save_csv("shared_backbone_metrics.csv", [
        ["指标", "对照方案", "shared-backbone", "变化"],
        ["AUROC", "0.7675", "0.7995", "+0.0320"],
        ["未知类别拒识率", "21.55%", "25.16%", "+3.61 个百分点"],
    ])
    kd_csv = save_csv("uncertainty_kd_metrics.csv", [
        ["类别划分", "对照方案", "uncertainty-KD", "结果"],
        ["划分 1", "29.63%", "22.90%", "下降"],
        ["划分 2", "30.55%", "26.80%", "下降"],
        ["划分 3", "28.95%", "29.48%", "上升"],
    ])
    save_csv("method_comparison_matrix.csv", [
        ["方法", "主要作用", "当前阶段结论"],
        ["scratch CNN", "建立不蒸馏的基础分类表征", "作为基础对照"],
        ["standard-KD", "迁移教师模型的分类语义", "作为蒸馏对照"],
        ["shared-backbone", "比较共享表征对未知检测的作用", "AUROC 与未知拒识率均改善"],
        ["uncertainty-KD", "按不确定性调整知识迁移", "机制已生效，但任务收益不稳定"],
    ])
    save_csv("evaluation_metric_mapping.csv", [
        ["任务环节", "评价目标", "主要指标"],
        ["已知类别分类", "保留基本分类能力", "分类准确率、已知类别效用"],
        ["未知样本拒识", "筛出训练类别之外的样本", "AUROC、未知类别拒识率"],
        ["新类别聚类", "形成稳定的潜在类别", "聚类准确率、候选覆盖与污染"],
    ])
    save_csv("experiment_protocol.csv", [
        ["实验条件", "当前设置"],
        ["数据集与类别划分", "CIFAR-100 60/40 与 semantic-hard 划分"],
        ["训练数据使用", "训练阶段仅使用已知类别标签"],
        ["阈值校准", "仅使用已知验证集，未知标签不参与校准"],
        ["对照方法", "scratch CNN、standard-KD、shared-backbone、uncertainty-KD"],
        ["重复实验", "三组不同类别划分"],
    ])
    save_csv("result_summary.csv", [
        ["方法与指标", "对照", "当前方案", "变化或结论"],
        ["shared-backbone AUROC", "0.7675", "0.7995", "+0.0320，三组方向一致"],
        ["shared-backbone 未知拒识率", "21.55%", "25.16%", "+3.61 个百分点，三组方向一致"],
        ["uncertainty-KD 未知拒识率均值", "29.71%", "26.39%", "均值下降 3.32 个百分点，但两降一升"],
    ])
    save_csv("uncertainty_kd_mean.csv", [
        ["统计范围", "对照均值", "uncertainty-KD 均值", "结论"],
        ["三组类别划分", "29.71%", "26.39%", "均值下降，但任务收益不稳定"],
    ])
    images = [
        make_shared_chart(), make_uncertainty_chart(), make_pipeline(), make_issue_chain(),
        make_evaluation_framework(), make_method_matrix(),
    ]
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_image_slide(prs, "图 1：开放环境下的新类发现实验流程", images[2])
    add_image_slide(prs, "图 2：shared-backbone 指标对比", images[0])
    add_image_slide(prs, "图 3：uncertainty-KD 三组类别划分对比", images[1])
    add_image_slide(prs, "图 4：当前问题定位与后续分析方向", images[3])
    add_image_slide(prs, "图 5：三项任务与评价指标的对应关系", images[4])
    add_image_slide(prs, "图 6：模型与方法对照矩阵", images[5])
    add_table_slide(prs, "表 1：shared-backbone 指标对比", [
        ["指标", "对照方案", "shared-backbone", "变化"],
        ["AUROC", "0.7675", "0.7995", "+0.0320"],
        ["未知类别拒识率", "21.55%", "25.16%", "+3.61 个百分点"],
    ])
    add_table_slide(prs, "表 2：uncertainty-KD 三组实验结果", [
        ["类别划分", "对照方案", "uncertainty-KD", "结果"],
        ["划分 1", "29.63%", "22.90%", "下降"],
        ["划分 2", "30.55%", "26.80%", "下降"],
        ["划分 3", "28.95%", "29.48%", "上升"],
    ])
    add_table_slide(prs, "表 3：任务环节与评价指标", [
        ["任务环节", "评价目标", "主要指标"],
        ["已知类别分类", "保留基本分类能力", "分类准确率、已知类别效用"],
        ["未知样本拒识", "筛出训练类别之外的样本", "AUROC、未知类别拒识率"],
        ["新类别聚类", "形成稳定的潜在类别", "聚类准确率、候选覆盖与污染"],
    ])
    add_table_slide(prs, "表 4：实验条件与数据使用规则", [
        ["实验条件", "当前设置"],
        ["数据集与类别划分", "CIFAR-100 60/40 与 semantic-hard 划分"],
        ["训练数据使用", "训练阶段仅使用已知类别标签"],
        ["阈值校准", "仅使用已知验证集，未知标签不参与校准"],
        ["对照方法", "scratch CNN、standard-KD、shared-backbone、uncertainty-KD"],
        ["重复实验", "三组不同类别划分"],
    ])
    add_table_slide(prs, "表 5：阶段性结果汇总", [
        ["方法与指标", "对照", "当前方案", "变化或结论"],
        ["shared-backbone AUROC", "0.7675", "0.7995", "+0.0320，三组方向一致"],
        ["shared-backbone 未知拒识率", "21.55%", "25.16%", "+3.61 个百分点，三组方向一致"],
        ["uncertainty-KD 未知拒识率均值", "29.71%", "26.39%", "均值下降，但两降一升"],
    ])
    add_table_slide(prs, "表 6：uncertainty-KD 均值对比", [
        ["统计范围", "对照均值", "uncertainty-KD 均值", "结论"],
        ["三组类别划分", "29.71%", "26.39%", "均值下降，但任务收益不稳定"],
    ])
    pptx_path = OUT / "中期汇报可视化素材.pptx"
    prs.save(pptx_path)
    (OUT / "素材说明.txt").write_text(
        "建议用法：图1放在总体研究方法页，图2和表1放在阶段性发现页，图3和表2放在uncertainty-KD结果位置，图4放在实验进展与问题页。\n"
        "图5适合放在评价体系或实验流程页，图6适合放在方法对照页。新增表3至表6用于补充评价指标、实验条件和结果汇总。\n"
        "PNG为独立图片，CSV为可编辑数据表，PPTX为可直接复制的素材汇总。\n",
        encoding="utf-8",
    )
    print(pptx_path)


if __name__ == "__main__":
    main()
