from Bio import pairwise2
from Bio.Seq import Seq
import pandas as pd

# 读取包含miRNA序列信息的Excel文件
df = pd.read_excel("mirna_sequences.xlsx")

# 定义一个函数，用于计算两个miRNA序列的相似性得分
def compute_similarity(seq1, seq2):
    # max_score = min(len(str(seq1)), len(str(seq2)))
    alignment = pairwise2.align.globalxx(Seq(str(seq1)), Seq(str(seq2)), one_alignment_only=True)
    # score = alignment[0].score / max_score
    score = alignment[0].score
    return score

# 创建一个空矩阵，用于存储相似性得分
n = len(df)
similarity_matrix = pd.DataFrame(index=df["miRNA"], columns=df["miRNA"], dtype=float)

# 计算相似性得分，并将其存储到矩阵中
for i in range(n):
    for j in range(i, n):
        mirna1 = df.loc[i, "miRNA"]
        mirna2 = df.loc[j, "miRNA"]
        seq1 = df.loc[i, "Sequence"]
        seq2 = df.loc[j, "Sequence"]
        score = compute_similarity(seq1, seq2)
        similarity_matrix.loc[mirna1, mirna2] = score
        similarity_matrix.loc[mirna2, mirna1] = score

# 将相似性矩阵保存到Excel文件中
similarity_matrix.to_excel("mirna_seq_similarity.xlsx")
