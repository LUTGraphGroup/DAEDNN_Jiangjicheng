from Bio import SeqIO
import pandas as pd

# 读取Excel表格
df = pd.read_excel("mirna-list.xlsx")

# 创建一个空字典，用于存储miRNA序列信息
mirna_sequences = {}

# 遍历Excel表格中的miRNA列表
for mirna_name in df["miRNA"]:
    # 从.fa文件中提取序列信息
    with open("mirna_sequences.fa") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record.id == mirna_name:
                mirna_sequences[mirna_name] = str(record.seq)
                break

# 将miRNA序列信息保存到Excel表格中
df["Sequence"] = df["miRNA"].map(mirna_sequences)
df.to_excel("mirna_sequences.xlsx", index=False)
