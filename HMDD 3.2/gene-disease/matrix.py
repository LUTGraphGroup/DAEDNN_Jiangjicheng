import pandas as pd
import numpy as np

# 读取基因和疾病关联数据文件
df = pd.read_excel("gene-disease association.xlsx")

# 选择基因和疾病列，并去重
df = df[["geneSymbol", "diseaseName"]]
df = df.drop_duplicates()

# 获取基因和疾病的唯一值列表
gene_names = df["geneSymbol"].unique()
disease_names = df["diseaseName"].unique()

# 构建基因-疾病关联矩阵
matrix = pd.crosstab(df["geneSymbol"], df["diseaseName"])

# 对疾病按照原始文件中的顺序进行排序
matrix = matrix.reindex(columns=disease_names)

# 将未出现在关联数据文件中的基因和疾病对应的关联值设为0
matrix = matrix.reindex(index=gene_names, columns=disease_names, fill_value=0)

# 将矩阵写入Excel文件
matrix.to_excel("gene-disease-association-matrix.xlsx")
