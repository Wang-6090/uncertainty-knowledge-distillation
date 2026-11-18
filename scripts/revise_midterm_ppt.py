from pathlib import Path
import os
import copy
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
# 该目录中有多个历史版本，采用算法顺序版原始文件的唯一大小作为稳定识别条件，
# 避免 Windows 控制台编码导致中文 glob 退化成“匹配所有 PPTX”。
source_candidates = [p for p in ROOT.glob("*.pptx") if p.stat().st_size == 317612]
if len(source_candidates) != 1:
    raise RuntimeError(f"无法唯一定位算法顺序版源文件：{source_candidates}")
SOURCE = source_candidates[0]
OUTPUT = ROOT / "最新版_基于不确定性知识蒸馏的新类发现方法中期汇报_逻辑与布局复核版.pptx"

NAVY = RGBColor(50, 74, 122)
BLUE = RGBColor(79, 115, 184)
LIGHT = RGBColor(247, 249, 253)
LINE = RGBColor(196, 209, 230)
TEXT = RGBColor(51, 51, 51)
MUTED = RGBColor(95, 108, 124)
WHITE = RGBColor(255, 255, 255)
PALE_BLUE = RGBColor(235, 242, 252)
PALE_GREEN = RGBColor(235, 247, 241)
GREEN = RGBColor(62, 133, 94)
PALE_ORANGE = RGBColor(253, 244, 231)
ORANGE = RGBColor(181, 112, 36)


def inch(v):
    return Inches(v)


def normalize_pptx_package(source):
    """按 presentation.xml 中的展示顺序重建 slide 文件，去除隐藏旧页和重复页文件。"""
    tmp = tempfile.NamedTemporaryFile(prefix="clean_pptx_", suffix=".pptx", delete=False)
    tmp.close()
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    pres_ns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    rid_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(source, "r") as zin:
        infos = zin.infolist()
        last = {}
        for info in infos:
            last[info.filename] = info
        rel_root = ET.fromstring(zin.read("ppt/_rels/presentation.xml.rels"))
        rid_to_target = {}
        for rel in rel_root.findall(f"{{{rel_ns}}}Relationship"):
            target = rel.attrib.get("Target", "")
            if target.startswith("slides/"):
                rid_to_target[rel.attrib["Id"]] = "ppt/" + target
        pres_root = ET.fromstring(zin.read("ppt/presentation.xml"))
        ordered_old = []
        sld_id_lst = pres_root.find(f"{{{pres_ns}}}sldIdLst")
        for sld in list(sld_id_lst):
            old = rid_to_target.get(sld.attrib.get(rid_attr))
            if old:
                ordered_old.append(old)
        old_to_new = {old: f"ppt/slides/slide{i}.xml" for i, old in enumerate(ordered_old, 1)}
        rel_old_to_new = {old.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels":
                          new.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
                          for old, new in old_to_new.items()}
        slide_re = re.compile(r"^ppt/slides/slide\d+\.xml$")
        rel_re = re.compile(r"^ppt/slides/_rels/slide\d+\.xml\.rels$")
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, info in last.items():
                if slide_re.match(name):
                    if name not in old_to_new:
                        continue
                    new_name = old_to_new[name]
                elif rel_re.match(name):
                    if name not in rel_old_to_new:
                        continue
                    new_name = rel_old_to_new[name]
                else:
                    new_name = name
                data = zin.read(info)
                if name == "ppt/_rels/presentation.xml.rels":
                    root = ET.fromstring(data)
                    for rel in list(root):
                        target = rel.attrib.get("Target", "")
                        if target.startswith("slides/"):
                            old = "ppt/" + target
                            if old not in old_to_new:
                                root.remove(rel)
                            else:
                                rel.set("Target", old_to_new[old].replace("ppt/", ""))
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                elif name == "[Content_Types].xml":
                    root = ET.fromstring(data)
                    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
                    for child in list(root):
                        if child.attrib.get("PartName", "").startswith("/ppt/slides/"):
                            root.remove(child)
                    for new in old_to_new.values():
                        ET.SubElement(root, f"{{{ct_ns}}}Override", {
                            "PartName": "/" + new,
                            "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
                        })
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                out_info = copy.copy(info)
                out_info.filename = new_name
                zout.writestr(out_info, data)
    return Path(tmp.name)


def remove_shape(slide, shape):
    slide.shapes._spTree.remove(shape._element)


def clear_after(slide, keep_count):
    for shape in list(slide.shapes)[keep_count:]:
        remove_shape(slide, shape)


def set_text(shape, text, size=12, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
             font="微软雅黑", valign=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = inch(0.08)
    tf.margin_right = inch(0.08)
    tf.margin_top = inch(0.04)
    tf.margin_bottom = inch(0.04)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(0)
        p.space_before = Pt(0)
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def add_text(slide, x, y, w, h, text, size=12, color=TEXT, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    set_text(shape, text, size, color, bold, align, valign=valign)
    return shape


def add_box(slide, x, y, w, h, title, body, title_color=WHITE,
            body_color=TEXT, fill=LIGHT, line=LINE, title_size=13,
            body_size=11.5, title_h=0.42, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    box = slide.shapes.add_shape(kind, inch(x), inch(y), inch(w), inch(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = line
    box.line.width = Pt(0.8)
    head = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                  inch(x), inch(y), inch(w), inch(title_h))
    head.fill.solid()
    head.fill.fore_color.rgb = title_color if title_color != WHITE else NAVY
    head.line.fill.background()
    set_text(head, title, title_size, WHITE, True, PP_ALIGN.LEFT)
    tx = add_text(slide, x + 0.10, y + title_h + 0.08, w - 0.20,
                  h - title_h - 0.12, body, body_size, body_color, False,
                  PP_ALIGN.LEFT, MSO_ANCHOR.TOP)
    return box, head, tx


def add_arrow(slide, x1, y1, x2, y2, color=BLUE, width=1.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, inch(x1), inch(y1), inch(x2), inch(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    return c


def style_existing(shape, size=None, color=None, bold=None):
    if not hasattr(shape, "text_frame"):
        return
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if size is not None:
                r.font.size = Pt(size)
            if color is not None:
                r.font.color.rgb = color
            if bold is not None:
                r.font.bold = bold


clean_source = normalize_pptx_package(SOURCE)
prs = Presentation(str(clean_source))

# 1.2：把高层方法改成真实执行顺序，避免把训练阶段和推理阶段混成一条关系。
slide = prs.slides[2]
set_text(slide.shapes[7], "1.2 总体研究方法", 22, NAVY, True)
set_text(slide.shapes[8], "项目先固定开放环境实验协议，再完成特征学习；模型输出用于未知样本筛选，候选样本随后进入聚类。", 13.5, MUTED)
set_text(slide.shapes[12], "训练阶段只使用已知类别标注；未知类别标签仅用于最终评价。", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "学习能够区分已知类别的特征表征，为后续未知判断提供基础。", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[20], "依据模型输出判断样本是否偏离已知类别的支持范围，并筛选未知候选。", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[24], "对未知候选的特征进行 PCA 降维和 KMeans 聚类，并统一评价结果。", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[11], "先固定开放环境协议", 12, WHITE, True)
set_text(slide.shapes[15], "再学习已知类别表征", 12, WHITE, True)
set_text(slide.shapes[19], "随后完成未知检测", 12, WHITE, True)
set_text(slide.shapes[23], "最后形成候选新类", 12, WHITE, True)
set_text(slide.shapes[28], "各模块的衔接顺序", 13, NAVY, True)
set_text(slide.shapes[29], "分类特征先用于未知样本筛选，筛选出的候选样本再进行 PCA 降维和 KMeans 聚类，最后统一评价。", 13.8, TEXT)
set_text(slide.shapes[30], "流程先保证未知候选的质量，再进行降维、聚类和全链路评价。", 11, MUTED)

# 1.1：压缩背景说明区域，给下方“项目任务链路”留出独立空间。
slide = prs.slides[1]
set_text(slide.shapes[10], "封闭集模型假设训练与测试类别一致；面对训练阶段未出现的新类别，模型容易将其误判为已知类别。项目聚焦未知检测与新类发现两个衔接环节。", 12.5, TEXT, False, valign=MSO_ANCHOR.TOP)
slide.shapes[10].top = inch(2.52)
slide.shapes[10].height = inch(0.62)
set_text(slide.shapes[15], "本项目的问题定义", 12, NAVY, True)
set_text(slide.shapes[18], "本项目的研究目标", 12, NAVY, True)
set_text(slide.shapes[23], "本项目的任务链路", 12, NAVY, True)
set_text(slide.shapes[19], "本项目需要完成的任务", 12, TEXT, True)
set_text(slide.shapes[20], "当前需要解决的关键问题", 12, TEXT, True)
set_text(slide.shapes[26], "学习已知类别的判别表征", 12, WHITE, True)
set_text(slide.shapes[27], "学习可区分已知类别的表征。", 9.5, TEXT, False, valign=MSO_ANCHOR.TOP)
slide.shapes[27].top = inch(4.38)
slide.shapes[27].height = inch(0.18)
set_text(slide.shapes[31], "判断样本是否偏离已知类别", 12, WHITE, True)
set_text(slide.shapes[32], "判断样本是否偏离已知范围。", 9.5, TEXT, False, valign=MSO_ANCHOR.TOP)
slide.shapes[32].top = inch(4.38)
slide.shapes[32].height = inch(0.18)
set_text(slide.shapes[36], "筛选未知候选并形成新类别簇", 12, WHITE, True)
set_text(slide.shapes[37], "将未知候选组织为新类别簇。", 9.5, TEXT, False, valign=MSO_ANCHOR.TOP)
slide.shapes[37].top = inch(4.38)
slide.shapes[37].height = inch(0.18)

# 2.1：明确分类主干、拒识编码器和不确定性信号的关系。
slide = prs.slides[3]
set_text(slide.shapes[8], "训练完成后，模型输出分类特征、拒识特征和不确定性信号，分别服务于分类、未知检测和结果评价。", 13.2, MUTED)
set_text(slide.shapes[24], "ResNet-18\n学习分类特征，供拒识分支初始化或对照使用", 11, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[26], "模型输出：分类路径与拒识路径分别产生结果", 13, NAVY, True)
set_text(slide.shapes[20], "知识蒸馏传递教师信息", 12.5, NAVY, True)
set_text(slide.shapes[30], "分别进入分类主干和拒识编码器", 11, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[35], "分类路径：ResNet-18 → 已知类别 logits 与分类特征\n拒识路径：拒识编码器 → 拒识特征与拒识分数\n教师不确定性用于 uncertainty-KD 加权；学生侧置信度、预测熵和不确定性头分数用于拒识评分对照", 10.0, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[42], "拒识评分、候选筛选、PCA 和 KMeans 按照后续流程依次执行。", 11, MUTED)
set_text(slide.shapes[13], "CIFAR-100\n已知类别带标注，未知类别不提供标注", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[17], "教师模型负责提供知识", 11.5, WHITE, True)
set_text(slide.shapes[18], "ResNet-34\n负责输出软标签与分类语义", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[23], "学生模型负责学习表征", 11.5, WHITE, True)
set_text(slide.shapes[24], "ResNet-18\n负责学习分类特征，并供拒识分支初始化或对照使用", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[25], "教师模型：提供知识\n学生模型：学习表征", 11, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[29], "同一测试样本分别进入两条路径", 10.5, WHITE, True)
set_text(slide.shapes[30], "样本分别进入分类主干和拒识编码器", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[12], "输入图像进入模型", 12, WHITE, True)
set_text(slide.shapes[23], "学生模型负责学习表征", 11.5, WHITE, True)
set_text(slide.shapes[29], "同一测试样本分别进入两条路径", 10.5, WHITE, True)
set_text(slide.shapes[30], "样本分别进入分类主干和拒识编码器", 10.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[34], "模型产生两类输出", 13, NAVY, True)
# 删除窄小的“后续处理”框，避免文字竖排；把模型输出框扩展为清晰的主输出区域。
for shape in [slide.shapes[idx] for idx in range(36, 42)]:
    remove_shape(slide, shape)
for idx, left, width in [(32, 3.45, 8.65), (33, 3.45, 8.65),
                         (34, 3.62, 8.30), (35, 3.76, 8.02)]:
    slide.shapes[idx].left = inch(left)
    slide.shapes[idx].width = inch(width)

# 2.2：删除会制造“连续算法流程”误读的顶部箭头，改成并列对照。
slide = prs.slides[4]
set_text(slide.shapes[7], "2.2 训练方案对照与输出", 22, NAVY, True)
set_text(slide.shapes[8], "三种训练方案并列对照；拒识编码器的初始化方式另做单因素比较。", 14, MUTED)
for shape in [slide.shapes[idx] for idx in [21, 22, 23, 24]]:
    remove_shape(slide, shape)
set_text(slide.shapes[11], "对照 A：Scratch CNN", 13.2, WHITE, True, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[15], "对照 B：standard-KD", 13.2, WHITE, True, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[19], "对照 C：uncertainty-KD", 12.8, WHITE, True, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[12], "不使用教师模型，只用已知类别标签训练，作为基础参照。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "加入教师模型，使用温度缩放后的 KL 散度传递软类别分布。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[20], "在 standard-KD 上加入不确定性权重，检验知识迁移是否更可靠。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[21], "训练完成后，结果进入下一环节", 12.5, NAVY, True)
set_text(slide.shapes[24], "训练得到的分类表征进入拒识模块", 11.5, WHITE, True)
set_text(slide.shapes[28], "训练得到的不确定性信号进入后续评分", 11.0, WHITE, True)
set_text(slide.shapes[30], "模型完成特征学习后\n进入拒识流程", 10.5, TEXT, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[24], "学生分类特征供拒识模块使用", 11.5, WHITE, True)
set_text(slide.shapes[25], "学生主干输出分类特征，拒识编码器基于该特征学习未知风险分数。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[28], "不确定性信号分别服务于训练和评分", 10.8, WHITE, True)
set_text(slide.shapes[29], "教师不确定性用于 uncertainty-KD 的蒸馏加权；分类熵和学生不确定性头分数可作为后续拒识评分的参考。", 10.3, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[30], "模型完成特征学习后\n进入未知检测", 10.5, TEXT, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[31], "特征学习完成后，模型依次进入拒识特征提取、NNPU 风险估计、阈值校准和候选筛选流程。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)

# 2.3：将步骤标题改为可直接讲述的动作短句。
slide = prs.slides[5]
set_text(slide.shapes[11], "① 先提取用于拒识的特征", 10.8, WHITE, True)
set_text(slide.shapes[15], "② 再初始化拒识器", 11.5, WHITE, True)
set_text(slide.shapes[19], "③ 估计未知风险", 11.5, WHITE, True)
set_text(slide.shapes[23], "④ 确定拒识阈值", 11.5, WHITE, True)
set_text(slide.shapes[27], "⑤ 筛选未知候选", 11.5, WHITE, True)
set_text(slide.shapes[12], "学生主干输出分类特征，供拒识编码器进一步形成拒识特征。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "scratch_cnn 从随机参数开始；shared-backbone 从学生主干初始化。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[20], "已知样本提供标签；混合发现池不提供标签。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[24], "只使用已知验证集确定拒识阈值。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[28], "超过拒识工作点的样本进入候选池。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)

# 2.4：将聚类步骤标题改为连续动作，避免只出现算法名词。
slide = prs.slides[6]
set_text(slide.shapes[11], "先确定候选池", 11.5, WHITE, True)
set_text(slide.shapes[15], "再提取候选特征", 11.5, WHITE, True)
set_text(slide.shapes[19], "随后用 PCA 降维", 11.0, WHITE, True)
set_text(slide.shapes[23], "再用 KMeans 聚类", 11.0, WHITE, True)
set_text(slide.shapes[27], "最后进行指标评价", 11.0, WHITE, True)
set_text(slide.shapes[12], "拒识通过的样本组成未知候选池。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "使用候选样本的拒识特征或投影特征。", 11.2, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[20], "PCA 压缩冗余维度，并保留主要变化方向。", 11.0, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[24], "KMeans 按特征相似性形成候选类别簇。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[28], "使用 ACC、NMI、ARI，并结合覆盖率与污染率分析。", 10.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[36], "先评价候选池质量", 12.5, NAVY, True)
set_text(slide.shapes[40], "再评价聚类质量", 12.5, NAVY, True)

# 2.5：从“方法介绍”改为“验证标准”，避免重复 2.2 和 2.3 的算法说明。
slide = prs.slides[7]
set_text(slide.shapes[8], "项目重点关注三件事：机制是否真正生效，局部指标是否改善，完整任务收益是否稳定。", 13.5, MUTED)
set_text(slide.shapes[12], "检验不确定性权重是否真正改变蒸馏过程，以及这种改变能否在多组类别划分中转化为稳定收益。", 11.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "检验 shared-backbone 的初始化能否改善拒识表征，并区分局部拒识收益与完整新类发现收益。", 11.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[20], "把分类、拒识、候选池和聚类放入同一评价链路，分别观察排序、工作点、覆盖、污染和聚类结果。", 11.8, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[22], "判断标准：机制真实执行、局部指标改善、完整任务收益稳定，三者分别核验。", 12.5, TEXT, True)

# 3.1：把拥挤的路线段落改成两栏，左边说明瓶颈，右边展示实验顺序。
slide = prs.slides[8]
clear_after(slide, 8)
set_text(slide.shapes[7], "3.1 当前问题与验证思路", 22, NAVY, True)
add_box(slide, 0.78, 1.92, 5.25, 4.65, "当前核心问题",
        "语义相近的新类别可能与已知类别共享相似特征。\n\n• 特征和拒识分数重叠，未知检测不稳定；\n• AUROC 的排序变化不一定转化为固定工作点的拒识收益；\n• 漏检与已知污染共同影响后续新类聚类。",
        fill=PALE_BLUE, line=LINE, title_size=14, body_size=12.5, title_h=0.48)

add_text(slide, 6.55, 1.88, 5.95, 0.30, "问题对应的验证思路", 15, NAVY, True)
route = [
    ("01", "固定协议与基线", "统一类别划分、训练设置和评价规则；完成 scratch CNN 与 standard-KD。"),
    ("02", "比较拒识表征", "对照 scratch CNN 与 shared-backbone，检查分类知识是否改善未知检测。"),
    ("03", "审计 uncertainty-KD", "核对不确定性权重、蒸馏损失和梯度是否真正参与优化。"),
    ("04", "分解任务误差", "结合候选覆盖、候选污染和聚类指标，定位完整新类发现的瓶颈。"),
]
ys = [2.22, 3.18, 4.14, 5.10]
for k, ((num, title, body), y) in enumerate(zip(route, ys)):
    if k < len(route) - 1:
        add_arrow(slide, 7.00, y + 0.76, 7.00, y + 0.98, color=LINE, width=1.0)
    circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, inch(6.62), inch(y), inch(0.72), inch(0.72))
    circ.fill.solid(); circ.fill.fore_color.rgb = NAVY; circ.line.fill.background()
    set_text(circ, num, 10.5, WHITE, True, PP_ALIGN.CENTER)
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, inch(7.48), inch(y - 0.02), inch(4.72), inch(0.76))
    card.fill.solid(); card.fill.fore_color.rgb = LIGHT; card.line.color.rgb = LINE
    add_text(slide, 7.66, y + 0.03, 1.62, 0.26, title, 11.5, NAVY, True)
    add_text(slide, 9.25, y + 0.02, 2.78, 0.58, body, 9.7, TEXT, False, valign=MSO_ANCHOR.TOP)
add_text(slide, 0.98, 6.78, 11.4, 0.28, "每一步同时记录排序指标、固定工作点指标以及候选池和聚类指标，避免只依据单一指标判断。", 11, MUTED, False, PP_ALIGN.CENTER)

# 3.2：保留四步卡片，但把表述改成“完成了什么”，并解释三组划分的含义。
slide = prs.slides[9]
set_text(slide.shapes[8], "已完成工作按“协议—模型—任务链路—结果核验”展开，形成可运行的实验闭环。", 13.5, MUTED)
slide.shapes[8].top = inch(1.88)
slide.shapes[8].height = inch(0.28)
set_text(slide.shapes[10], "先统一实验协议", 12.5, NAVY, True)
set_text(slide.shapes[14], "再建立模型对照", 12.5, NAVY, True)
set_text(slide.shapes[18], "打通识别与聚类", 12.5, NAVY, True)
set_text(slide.shapes[22], "完成对照与训练审计", 12.5, NAVY, True)
set_text(slide.shapes[11], "完成 CIFAR-100 中 60 个已知类别、40 个未知类别的划分；固定数据增强、训练/验证/测试隔离和评价规则，为方法对照提供统一条件。", 11.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[15], "完成 scratch CNN、standard-KD、shared-backbone 和 uncertainty-KD 的训练入口；统一输入、训练设置和输出表征。", 11.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[19], "完成分类、不确定性估计、拒识器训练、NNPU 风险估计、候选筛选和 PCA+KMeans 聚类；未知标签只用于最终评价。", 11.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[23], "在三组预先固定的语义相近类别划分中记录分类、AUROC、未知拒识率和聚类指标；同时审计 uncertainty-KD 是否真正参与优化。", 11.5, TEXT, False, valign=MSO_ANCHOR.TOP)

# 3.3：把近期结果写成可直接比较的原始值—当前值，并说明 uncertainty-KD 的比较对象。
slide = prs.slides[10]
set_text(slide.shapes[9], "实验设置：CIFAR-100 semantic-hard 60/40，三套类别划分。\nscratch CNN：AUROC 0.7675；拒识率 21.55%\nshared-backbone：AUROC 0.7995；拒识率 25.16%\n相较 scratch CNN，两个指标分别高 0.0320 和 3.61 个百分点。", 10.7, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[12], "uncertainty-KD 已参与训练，审计显示权重、蒸馏损失和梯度均非零。\n相较 standard-KD，三组划分的拒识率两组下降、一组上升，尚未形成稳定增益。", 11.5, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[10], "shared-backbone 对照结果", 12.2, WHITE, True)
set_text(slide.shapes[13], "uncertainty-KD 的稳定性", 12.0, WHITE, True)
set_text(slide.shapes[15], "当前已完成分类—未知检测—候选聚类闭环。共享表征在未知检测环节已有一致的局部证据；完整新类发现仍需结合候选覆盖、候选污染和聚类结果继续复核。", 13, TEXT, False, valign=MSO_ANCHOR.TOP)
set_text(slide.shapes[16], "本阶段综合判断", 12.5, TEXT, True)
slide.shapes[15].top = inch(6.02)
slide.shapes[15].height = inch(1.12)
slide.shapes[16].top = inch(6.12)
# 结果卡片整体下移，避免与页面标题形成压叠。
for idx in [8, 11]:
    slide.shapes[idx].top = inch(1.90)
    slide.shapes[idx].height = inch(3.95)
slide.shapes[9].top = inch(2.40)
slide.shapes[9].height = inch(1.35)
slide.shapes[9].top = inch(2.42)
slide.shapes[10].top = inch(2.02)
slide.shapes[10].height = inch(0.32)
slide.shapes[12].top = inch(2.40)
slide.shapes[12].top = inch(2.42)
slide.shapes[13].top = inch(2.02)
slide.shapes[13].height = inch(0.32)
slide.shapes[17].top = inch(3.62)
slide.shapes[17].height = inch(2.15)
slide.shapes[15].left = inch(3.20)
slide.shapes[15].top = inch(6.16)
slide.shapes[15].width = inch(8.65)
slide.shapes[15].height = inch(0.86)
slide.shapes[16].top = inch(6.18)
slide.shapes[16].height = inch(0.30)

# 3.4：用四阶段横向进度替代折返式时间轴，突出当前节点；同时彻底移除旧时间轴元素。
slide = prs.slides[11]
keep = [slide.shapes[i] for i in [0, 1, 2, 3, 4, 15, 16]]
for shape in list(slide.shapes):
    if shape not in keep:
        remove_shape(slide, shape)
add_text(slide, 0.82, 1.84, 11.7, 0.22, "项目按“准备—实现—复核—收束”推进；当前处于核心模块实现完成、结果复核展开的节点。", 13.5, MUTED)
stages = [
    ("已完成工作", "2026.06—09", "研究准备与实验协议", "完成问题定义、类别划分、预处理和基础训练环境，建立可复现实验入口。", NAVY, PALE_BLUE),
    ("当前正在进行", "2026.10—11", "模型与首轮实验", "完成教师—学生训练、拒识分支、候选聚类和首轮对照，正在复核结果。", BLUE, PALE_BLUE),
    ("下一步安排", "2026.12—2027.03", "结果复核与框架完善", "补充多划分和消融实验，分析候选覆盖、候选污染与支持区域边界。", GREEN, PALE_GREEN),
    ("后续收束", "2027.04—06", "综合实验与结题", "完成图表整理、综合结论、结题材料定稿和项目验收准备。", ORANGE, PALE_ORANGE),
]
xs = [0.58, 3.78, 6.98, 10.18]
for i, ((tag, date, title, body, head_color, fill), x) in enumerate(zip(stages, xs)):
    if i < len(stages) - 1:
        add_arrow(slide, x + 2.62, 3.50, xs[i+1] - 0.13, 3.50, color=LINE, width=1.0)
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, inch(x), inch(2.08), inch(2.58), inch(3.78))
    box.fill.solid(); box.fill.fore_color.rgb = fill; box.line.color.rgb = LINE
    head = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, inch(x), inch(2.08), inch(2.58), inch(0.48))
    head.fill.solid(); head.fill.fore_color.rgb = head_color; head.line.fill.background()
    set_text(head, tag, 13, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, x + 0.12, 2.70, 2.34, 0.28, date, 10.5, head_color, True, PP_ALIGN.CENTER)
    add_text(slide, x + 0.14, 3.15, 2.30, 0.60, title, 13, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, x + 0.16, 4.00, 2.26, 1.40, body, 10.8, TEXT, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP)
add_text(slide, 0.88, 6.35, 11.55, 0.32, "当前节点的工作重点：复核模型对照结果，并将拒识表现与候选池质量、聚类稳定性联系起来分析。", 11.5, MUTED, False, PP_ALIGN.CENTER)

# 4.1：将三个后续重点统一为同一基线的卡片，避免标题和正文高低错位。
slide = prs.slides[12]
clear_after(slide, 7)
add_text(slide, 0.82, 1.84, 11.70, 0.22,
         "下一阶段围绕三个相互衔接的问题推进：先分析表征边界，再检查候选池质量，最后复核蒸馏收益。",
         13.5, MUTED)
followups = [
    ("语义相近类别", "分析新类别与已知类别的特征和拒识分数重叠，定位持续漏检的类别，并比较不同划分下的边界变化。", NAVY, PALE_BLUE),
    ("候选池质量", "分别记录未知覆盖率与已知污染率，区分漏检和误检来源，分析候选池质量对后续聚类的影响。", GREEN, PALE_GREEN),
    ("蒸馏收益稳定性", "复核不确定性来源、权重分布及跨划分结果，避免仅依据单个指标调整，并决定是否纳入默认方案。", ORANGE, PALE_ORANGE),
]
for x, (title, body, head_color, fill) in zip([0.72, 4.48, 8.24], followups):
    add_box(slide, x, 2.10, 3.52, 3.30, title, body,
            fill=fill, line=LINE, title_color=head_color,
            title_size=13.5, body_size=12.0, title_h=0.50)
add_text(slide, 0.90, 5.78, 11.50, 0.30,
         "每项改进均采用单因素对照，并同时观察拒识指标、候选池质量和聚类结果。",
         11.5, MUTED, False, PP_ALIGN.CENTER)

# 5.1：提升到项目整体结论，避免重复 3.3 的具体阶段性数据。
slide = prs.slides[13]
set_text(slide.shapes[8], "项目已完成从开放环境任务定义、数据协议、教师—学生训练，到分类、未知检测和候选聚类的实验链路，并建立了统一评价与训练审计方法。\n\n阶段性判断是：shared-backbone 在未知检测环节形成了局部改善证据；uncertainty-KD 机制已执行，但收益尚未稳定转化到完整任务。\n\n下一阶段将围绕语义相近类别的表征边界、候选池质量和全链路稳定性继续验证。", 15.5, TEXT, False, valign=MSO_ANCHOR.TOP)

# 处理模板中横线因浮点换算产生的极小越界，避免导出时被裁掉边缘像素。
for current_slide in prs.slides:
    for shape in current_slide.shapes:
        if shape.left >= 0 and shape.left + shape.width > prs.slide_width:
            shape.width = prs.slide_width - shape.left

prs.save(str(OUTPUT))
clean_source.unlink(missing_ok=True)
clean_output = normalize_pptx_package(OUTPUT)
os.replace(clean_output, OUTPUT)
print(OUTPUT)
