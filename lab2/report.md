# **机器学习概论 实验二：K-MEANS聚类算法 实验报告**
> 李文赢 计05 2020080108

## **实验简介**
本实验目标是实现 K-MEANS 聚类算法，并在真实数据集上进行评测

## **代码设计思路**
- 首先通过 `get_dataset` 函数加载 MNIST 数据集，并返回前 N 张图像及其对应的标签
- 然后通过 `K-MEANS` 类进行 K-MEANS 计算
  - 初始化 KMeans 类，设置**聚类数量为 30**、最大迭代次数为 100 和随机种子
  - `fit` 函数：拟合 KMeans 模型到给定数据上。**随机初始化 K 个聚类中心**，通过迭代的方式，将每个数据点**分配到距离它最近的聚类中心**，并更新聚类中心的位置，然后迭代更新聚类中心**直到收敛或达到最大迭代次数**
  - `_assign_clusters` 函数：计算每个数据点到每个聚类中心的距离，并将每个数据点分配到距离它最近的聚类中心，**使用欧氏距离来衡量样本之间的距离**
- `calculate_accuracy` 计算聚类的准确率，比较每个聚类中的样本标签与多数投票预测的标签，并计算正确预测的样本比例
- `visualize_clusters` 使用 t-SNE 技术将高维数据降维到二维空间，并将聚类结果可视化出来，每个数据点在二维空间中的位置代表了它们的相似性
- `evaluate_clustering` 函数用于评估聚类结果的质量，计算了聚类结果的轮廓系数，衡量了聚类的紧密度和分离度


## **模型评估结果**
当图片数量设定为 N=100，聚类设定数量为 K=30 时，得到的模型评价是：
**准确性：**
```python
K-Means Clustering Accuracy: 79.00%
```
**可视化聚类结果**
通过函数 `plot_result` 展示每个聚类的标签以及显示每个聚类的一个样本图像
- 使用 `np.where(labels == i)[0]` 找到属于该聚类的所有数据点的索引
- 从当前聚类中随机选择一个数据点的索引
- 在相应的子图中绘制图像，设置标题为聚类的标签，并在图像下方写入预测的标签

![img](/Users/nip/Desktop/清华/2024大四春季学期/机器学习概论/Exp2/lab2/img/result.png)
从图中可以看到 Label 为真实数值，prediction 为通过算法预测数值，而且每个聚类的值为如下：

```python
Cluster 0: Labels: tensor([0, 6, 6])
Cluster 1: Labels: tensor([5, 5, 4, 4, 5])
Cluster 2: Labels: tensor([1])
Cluster 3: Labels: tensor([4, 9])
Cluster 4: Labels: tensor([3, 3, 3])
Cluster 5: Labels: tensor([6, 6, 6])
Cluster 6: Labels: tensor([2, 9, 7, 9, 9])
Cluster 7: Labels: tensor([9, 2])
Cluster 8: Labels: tensor([3, 3, 2, 3, 3, 3])
Cluster 9: Labels: tensor([5])
Cluster 10: Labels: tensor([6, 6])
Cluster 11: Labels: tensor([3])
Cluster 12: Labels: tensor([6, 6, 6])
Cluster 13: Labels: tensor([8, 9, 9, 7])
Cluster 14: Labels: tensor([4, 6])
Cluster 15: Labels: tensor([9])
Cluster 16: Labels: tensor([2, 2, 0, 2])
Cluster 17: Labels: tensor([1, 1, 8, 1, 1, 1, 1, 1])
Cluster 18: Labels: tensor([3])
Cluster 19: Labels: tensor([8])
Cluster 20: Labels: tensor([8, 8, 8, 8, 8])
Cluster 21: Labels: tensor([0, 0, 0])
Cluster 22: Labels: tensor([4, 7, 9, 4, 7, 4])
Cluster 23: Labels: tensor([1, 7, 7])
Cluster 24: Labels: tensor([0, 0, 0, 0, 0, 0, 0, 0])
Cluster 25: Labels: tensor([7, 7, 9, 7])
Cluster 26: Labels: tensor([1])
Cluster 27: Labels: tensor([7])
Cluster 28: Labels: tensor([4, 4, 5, 4, 4])
Cluster 29: Labels: tensor([1, 1, 1, 9, 1, 3])
```

**可视化高维数据**
通过函数 `visualize_clusters` 用t-SNE 方法对数据进行降维，并可视化聚类结果
- t-SNE 方法对原始数据 data 进行降维，将其转换为二维空间中的点集，设置 n_components=2 表示降维到二维空间
- colorbar 显示了不同颜色与聚类标签之间的对应关系

![img](/Users/nip/Desktop/清华/2024大四春季学期/机器学习概论/Exp2/lab2/img/tsne.png)


**评估聚类结果的质量**
在 `evaluate_clustering` 计算了轮廓系数（Silhouette Score），轮廓系数是一种衡量聚类质量的指标，同时考虑了聚类的紧密度和分离度，计算得到的轮廓系数 `silhouette_avg` 表示所有数据点的平均轮廓系数

```python
Silhouette Score: 0.020955074578523636
```

## **实验分析**
- 在选择聚类数量时，发现很难确定最佳的聚类数量。不同的聚类数量可能会导致不同的聚类结果，最终选择设定为 30
- 在初始化聚类中心时，发现初始点的选择可能会影响最终的聚类结果和收敛性
- **可视化结果分析：** 一些聚类结果并不明显，即不同聚类之间的界限模糊，难以区分。因为数据本身的特点导致的，例如数据点之间的相似性较高，或者是因为聚类算法的限制导致的，例如某些聚类方法对于非凸形状的聚类较为困难。