from __future__ import annotations

from pathlib import Path
from shutil import copy2

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(
    p for p in ROOT.glob("*中期进展报告_模型内容补充版.docx")
    if not p.name.startswith("~$")
)
OUTPUT = ROOT / "基于不确定性知识蒸馏的新类发现方法中期进展报告_模型细节补充版.docx"


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

    content = doc.tables[2].cell(0, 0).text
    old = (
        "5.构建教师—学生模型并提出不确定性感知知识蒸馏策略。教师模型采用 ResNet-34，"
        "提供较充分的分类语义和软标签；学生模型采用 ResNet-18，学习可迁移特征，并在主干后"
        "设置分类头、不确定性估计头和拒识分支，分别服务于已知类别分类、预测不确定性量化和"
        "未知样本筛选。standard-KD 和 uncertainty-KD 用于比较普通知识迁移与不确定性感知知识"
        "迁移的作用。"
    )
    new = (
        "5.构建教师—学生模型并提出不确定性感知知识蒸馏策略。教师模型采用 ResNet-34，"
        "提供较充分的分类语义和软标签；学生模型采用 ResNet-18，学习可迁移特征，并在主干后"
        "设置分类头、不确定性估计头和拒识分支，分别服务于已知类别分类、预测不确定性量化和"
        "未知样本筛选。standard-KD 使用温度缩放后的 KL 散度传递教师的软类别分布；"
        "uncertainty-KD 在此基础上根据教师不确定性调整逐样本蒸馏权重，比较普通知识迁移与"
        "不确定性感知知识迁移的作用。"
    )
    if old not in content:
        raise RuntimeError("未找到研究主要内容的模型段落。")
    set_cell_text(doc.tables[2].cell(0, 0), content.replace(old, new))

    work = doc.tables[3].cell(0, 0).text
    old3 = (
        "3. 完成教师模型和学生模型的基础训练程序。教师模型采用 ResNet-34，负责学习已知类别的分类语义并输出软标签；学生模型采用 ResNet-18，通过知识蒸馏学习可迁移特征。学生模型主干后设置分类头、不确定性估计头和拒识分支：分类头用于已知类别预测，不确定性估计头通过预测置信度和预测熵衡量样本可靠程度，拒识分支用于判断样本是否偏离已知类别支持区域。该结构为后续未知检测和新类别聚类提供统一的特征表示。"
    )
    new3 = (
        "3. 完成教师模型和学生模型的基础训练程序。教师模型采用 ResNet-34，负责学习已知类别的分类语义并输出软标签；学生模型采用 ResNet-18，通过知识蒸馏学习可迁移特征。学生模型主干后设置分类头、不确定性估计头和拒识分支：分类头用于已知类别预测，不确定性估计头输出独立的不确定性分数，分类概率还可计算最大置信度和预测熵，拒识分支用于判断样本是否偏离已知类别支持区域。当前数据流为输入图像—学生主干特征—分类、不确定性与拒识输出—候选筛选—PCA 降维和 KMeans 聚类。该结构为后续未知检测和新类别聚类提供统一的特征表示。"
    )
    if old3 not in work:
        raise RuntimeError("未找到已开展工作的模型段落。")
    work = work.replace(old3, new3)

    old4 = (
        "4. 完成标准知识蒸馏和不确定性知识蒸馏模块。standard-KD 作为基础对照，将教师模型的软类别信息传递给学生模型；uncertainty-KD 根据样本不确定性调整蒸馏权重，使知识迁移强度与样本预测可靠性相匹配，用于检验不确定性是否能够改善学生模型对开放环境的适应能力。对蒸馏权重、蒸馏损失和梯度的检查表明，两种蒸馏路径均已在程序中实际执行；但 uncertainty-KD 在不同重复实验中的任务结果有升有降，尚未形成稳定收益。对蒸馏权重、蒸馏损失和梯度的检查表明，两种蒸馏路径均已在程序中实际执行；但 uncertainty-KD 在不同重复实验中的任务结果有升有降，尚未形成稳定收益。"
    )
    new4 = (
        "4. 完成标准知识蒸馏和不确定性知识蒸馏模块。standard-KD 使用温度缩放后的 KL 散度，将教师模型的软类别信息传递给学生模型；uncertainty-KD 在逐样本 KL 损失上乘以由教师不确定性计算的权重，教师越确定，知识迁移约束越强。这里的教师不确定性权重用于 KD，不等同于学生不确定性头输出的分数；学生不确定性头、分类预测熵是否用于拒识，则由具体评分设置决定。对蒸馏权重、蒸馏损失和梯度的检查表明，两种蒸馏路径均已在程序中实际执行；但 uncertainty-KD 在不同重复实验中的任务结果有升有降，尚未形成稳定收益。"
    )
    if old4 not in work:
        raise RuntimeError("未找到已开展工作的知识蒸馏段落。")
    work = work.replace(old4, new4)

    old5 = (
        "5. 完成开放集识别和独立拒识分支。针对普通分类器容易把新类别样本纳入已知类别的问题，增加独立拒识分支，利用已知样本和无标签混合发现池估计已知类别支持区域。采用 NNPU 风险估计，是为了避免把混合发现池中的样本直接当作未知类别标注；采用 shared-backbone 初始化，是为了改善 scratch CNN 分支对学生模型特征的利用。三组不同类别划分的重复实验显示，该配置的拒识效果较好，但语义相近新类别与已知类别之间的重叠仍未完全解决。"
    )
    new5 = (
        "5. 完成开放集识别和独立拒识分支。针对普通分类器容易把新类别样本纳入已知类别的问题，增加独立拒识分支。当前 shared-backbone 的具体含义是：拒识编码器复制学生分类主干的结构和参数进行初始化，随后作为独立编码器继续训练，分类主干和拒识编码器并不持续共用同一组参数。拒识器使用有标签的已知样本和含有已知、未知样本的未标注混合发现池；NNPU 风险估计将已知样本作为可靠的已知侧样本，将混合池作为未标注混合分布，扣除其中可能来自已知分布的风险并进行非负截断，从而学习未知分数，而不是把混合池直接标为未知。随后根据拒识评分和已知验证集校准的工作点筛选候选。三组不同类别划分的重复实验显示，该配置的拒识效果较好，但语义相近新类别与已知类别之间的重叠仍未完全解决。"
    )
    if old5 not in work:
        raise RuntimeError("未找到已开展工作的拒识段落。")
    work = work.replace(old5, new5)

    old6 = (
        "6. 完成未知样本筛选和新类别聚类流程。使用已知验证集确定拒识工作点，将偏离已知类别支持区域的样本作为候选新类别样本；对候选特征进行 PCA 降维，并使用 KMeans 组织候选样本。"
    )
    new6 = (
        "6. 完成未知样本筛选和新类别聚类流程。数据流按照“学生模型输出—拒识评分—候选筛选—PCA 降维—KMeans 聚类”的顺序执行。使用已知验证集确定拒识工作点，将偏离已知类别支持区域的样本作为候选新类别样本；对候选特征进行 PCA 降维，并使用 KMeans 组织候选样本。"
    )
    if old6 not in work:
        raise RuntimeError("未找到已开展工作的候选聚类段落。")
    work = work.replace(old6, new6)
    set_cell_text(doc.tables[3].cell(0, 0), work)

    result = doc.tables[4].cell(0, 0).text
    result = result.replace(
        "建立 ResNet-34 教师模型与 ResNet-18 学生模型，并完成学生模型分类头、不确定性估计头和拒识分支的实现；同时完成 standard-KD、uncertainty-KD、未知拒识、候选筛选和新类聚类流程",
        "建立 ResNet-34 教师模型与 ResNet-18 学生模型，并完成学生模型分类头、不确定性估计头和拒识分支的实现；明确 standard-KD 与 uncertainty-KD 的逐样本蒸馏差异，完成 NNPU 拒识、候选筛选和新类聚类流程"
    )
    set_cell_text(doc.tables[4].cell(0, 0), result)

    doc.core_properties.title = "基于不确定性知识蒸馏的新类发现方法中期进展报告（模型细节补充版）"
    doc.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
