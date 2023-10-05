import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

matrix = pd.read_excel("gd-matrix-noheader.xlsx")
# 使用NMF提取特征
# model = NMF(n_components=374)
# features = model.fit_transform(matrix)
# print(model)
# print('-----------------')
# print(features)
# np.savetxt("1", features, delimiter="\t")
def sim_simpson(x, y):
    """
    计算基于Simpson相似性的两个向量之间的相似性
    """
    intersection = np.count_nonzero(np.logical_and(x > 0, y > 0))
    # return intersection / np.minimum(np.count_nonzero(x), np.count_nonzero(y))
    denominator = np.minimum(np.count_nonzero(x), np.count_nonzero(y))
    # intersection = np.count_nonzero(x == y)
    # denominator = len(x)
    if denominator == 0:
        return 0.0000
    else:
        return intersection / denominator
# 计算疾病相似性矩阵
n_diseases = matrix.shape[1]
similarity_matrix = np.zeros((n_diseases, n_diseases))
for i in range(n_diseases):
    for j in range(i, n_diseases):
        similarity_matrix[i, j] = sim_simpson(matrix.iloc[:, i], matrix.iloc[:, j])
        similarity_matrix[j, i] = similarity_matrix[i, j]

# 将相似性矩阵保存为CSV文件
np.savetxt("disease_similarity_matrix.txt", similarity_matrix, delimiter="\t", fmt='%.4f')