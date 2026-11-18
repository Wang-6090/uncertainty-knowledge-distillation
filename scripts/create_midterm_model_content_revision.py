from __future__ import annotations

from pathlib import Path
from shutil import copy2

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(
    p for p in ROOT.glob("*中期进展报告.docx")
    if "模型内容补充版" not in p.name
)
OUTPUT = ROOT / "基于不确定性知识蒸馏的新类发现方法中期进展报告_模型内容补充版.docx"


def set_cell_text(cell, text: str, size: float = 10.5) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "宋体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(size)


def main() -> None:
    copy2(SOURCE, OUTPUT)
    doc = Document(str(OUTPUT))

    # The research-content section stays aligned with the proposal, while the
    # model structure is made explicit for the midterm implementation stage.
    content_cell = doc.tables[2].cell(0, 0)
    content_text = content_cell.text
    old_content = (
        "5.提出不确定性感知知识蒸馏策略，将教师模型学到的语义知识与判别信息自适应迁移到学生模型，在提升模型精度的同时增强对开放环境的鲁棒性。"
    )
    new_content = (
        "5.构建教师—学生模型并提出不确定性感知知识蒸馏策略。教师模型采用 ResNet-34，"
        "提供较充分的分类语义和软标签；学生模型采用 ResNet-18，学习可迁移特征，并在主干后"
        "设置分类头、不确定性估计头和拒识分支，分别服务于已知类别分类、预测不确定性量化和"
        "未知样本筛选。standard-KD 和 uncertainty-KD 用于比较普通知识迁移与不确定性感知知识"
        "迁移的作用。"
    )
    if old_content not in content_text:
        raise RuntimeError("未找到研究主要内容中的知识蒸馏原文，未执行替换。")
    set_cell_text(content_cell, content_text.replace(old_content, new_content))

    work_cell = doc.tables[3].cell(0, 0)
    work_text = work_cell.text
    old_model_work = (
        "3. 完成教师模型和学生模型的基础训练程序。教师模型用于提供类别判别信息，学生模型使用已知类别标注进行分类学习，为后续知识蒸馏、未知检测和新类别聚类提供统一的特征表示。"
    )
    new_model_work = (
        "3. 完成教师模型和学生模型的基础训练程序。教师模型采用 ResNet-34，负责学习已知类别"
        "的分类语义并输出软标签；学生模型采用 ResNet-18，通过知识蒸馏学习可迁移特征。学生"
        "模型主干后设置分类头、不确定性估计头和拒识分支：分类头用于已知类别预测，不确定性"
        "估计头通过预测置信度和预测熵衡量样本可靠程度，拒识分支用于判断样本是否偏离已知"
        "类别支持区域。该结构为后续未知检测和新类别聚类提供统一的特征表示。"
    )
    if old_model_work not in work_text:
        raise RuntimeError("未找到已开展工作中的模型原文，未执行替换。")
    work_text = work_text.replace(old_model_work, new_model_work)

    old_kd_work = (
        "4. 完成标准知识蒸馏和不确定性知识蒸馏模块。standard-KD 作为基础对照，将教师模型的软类别信息传递给学生模型；uncertainty-KD 根据样本不确定性调整蒸馏权重，用于检验不确定性是否能够改善学生模型对开放环境的适应能力。"
    )
    new_kd_work = (
        "4. 完成标准知识蒸馏和不确定性知识蒸馏模块。standard-KD 作为基础对照，将教师模型"
        "的软类别信息传递给学生模型；uncertainty-KD 根据样本不确定性调整蒸馏权重，使知识"
        "迁移强度与样本预测可靠性相匹配，用于检验不确定性是否能够改善学生模型对开放环境"
        "的适应能力。对蒸馏权重、蒸馏损失和梯度的检查表明，两种蒸馏路径均已在程序中实际"
        "执行；但 uncertainty-KD 在不同重复实验中的任务结果有升有降，尚未形成稳定收益。"
    )
    if old_kd_work not in work_text:
        raise RuntimeError("未找到已开展工作中的知识蒸馏原文，未执行替换。")
    set_cell_text(work_cell, work_text.replace(old_kd_work, new_kd_work))

    result_cell = doc.tables[4].cell(0, 0)
    set_cell_text(
        result_cell,
        "截至中期，项目已完成开放环境下新类发现任务定义、CIFAR-100 类别划分与数据预处理，"
        "建立 ResNet-34 教师模型与 ResNet-18 学生模型，并完成学生模型分类头、不确定性估计头"
        "和拒识分支的实现；同时完成 standard-KD、uncertainty-KD、未知拒识、候选筛选和新类"
        "聚类流程，形成分类、拒识和聚类评价体系。多组对照实验表明，shared-backbone 在不同"
        "类别划分中均改善了 AUROC 和未知类别拒识率。"
    )

    doc.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期进展报告（模型内容补充版）"
    doc.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
