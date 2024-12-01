# **机器学习概论 实验一：朴素贝叶斯分类器 实验报告**
> 李文赢 计05 2020080108

## **实验简介**
本实验目标是实现一个朴素贝叶斯分类器并在真实数据集上进行评测

## **代码设计思路**
### **数据集预处理**
`dataset(train_prob, random_seed=None)` 函数将数据集分割成训练集和测试集
- 两个参数：`train_prob` 是训练集所占比例，`random_seed` 是随机数种子，用于实现结果的可复现性
- 首先从文件中读取标签数据
- 然后对数据进行了随机打乱
- 将数据集划分成训练集和测试集
- 将训练集和测试集的路径分别写入到文件中，分别命名为 `trainset` 和 `testset`

### **训练过程**
`train(sample_rate=1)` 函数用于训练分类器，创建了表示单词频率和类别频率的字典
- 接受一个参数 sample_rate，用于控制训练集的采样率
- `word_freq` 字典用于存储每个类别（垃圾邮件和非垃圾邮件）中单词的频率
  - 格式为 `{类别: {单词: 频率, ...}}`
- `total_freq` 字典用于存储每个类别的总词频
- 首先读取训练集中的每一行，对于每个文件根据 `sample_rate` 的值来决定是否要训练该文件
- 对于选定的文件，进行解析标签和文件路径，然后读取文件内容，并提取其中的单词
- 更新 `word_freq` 字典和 `total_freq` 字典，以记录单词的频率和类别的总词频
- 将这些频率信息保存到 JSON 文件中，分别命名为 `word_freq.json` 和 `total_freq.json`

### **测试过程**
`evaluate_model()` 函数用于评估模型的性能，使用五折交叉验证方法来评估模型在测试数据集上的表现
- `create_folds` 函数用于创建交叉验证的折叠，将数据集分成 k 折，其中每一折都是一个子数据集
- 首先加载测试数据集，并读取频率信息
- 使用 `create_folds` 函数将测试数据集分成五个折叠
- 对于每个折叠，代码计算了准确率、精确度、召回率和 F1 分数
- 在每个折叠内部，对于每个文件，代码计算了该文件属于每个类别的概率，并选择概率最大的类别作为预测结果
- 若采用了 Laplace 平滑，增加一个小的平滑因子 `l_index` 来避免零概率的问题
  - 对于每个单词，分子部分表示在类别 y 中单词出现的次数（加上平滑因子）
  - 分母部分表示类别 y 中所有单词的总数（加上平滑因子乘以单词表的大小）
  - 确保即使某个单词在训练集中没有出现过，也不会导致概率为零
- 计算每个折叠的准确率、精确度、召回率和 F1 分数，并存储在相应的列表中
- 最后，计算所有折叠的平均准确率、平均精确度、平均召回率和平均 F1 分数


## **模型评估结果**
平均准确率、平均精确度、平均召回率和平均 F1 分数结果如下：

在参数如下情况可以得到的结果
```python
# 参数
train_prob = 0.9
random_seed = 1
sample_rate = 1
laplace = 1
# 结果
Average Accuracy: 0.9920697659828093
Average Precision: 1.0
Average Recall: 0.9920697659828093
Average F1 Score: 0.9960185666568397
```

Accuracy 准确率为 99.2%，Precision 精确度为 100%，Recall 召回率为 99.2%，F1 分数为 99.6%，因此从数据得出了明显的结论是本模型的性能是非常高


## **实验分析**
### **问题1: 训练集大小对性能的影响**
分别采样 5%, 50% 和 100% 的训练集数据进行训练得到的结果

```python
Sample Rate:  0.05
Average Accuracy: 0.9413200326243805
Average Precision: 1.0
Average Recall: 0.9413200326243805
Average F1 Score: 0.9697507183647908
```
```python
Sample Rate:  0.5
Average Accuracy: 0.981236798628103
Average Precision: 1.0
Average Recall: 0.981236798628103
Average F1 Score: 0.9905176634844309
```
```python
Sample Rate:  1
Average Accuracy: 0.9849342284124892
Average Precision: 1.0
Average Recall: 0.9849342284124892
Average F1 Score: 0.9924051852571079
```

从上述的结果可以观察到，增加训练集大小可以提升分类器的性能，但是当训练集较小时，可能会出现较高的偏差，影响性能表现。随着训练集大小的增加，模型的性能表现也会相应提升，但增加训练集的边际收益会逐渐减小。


### **问题2: 零概率问题**
当某些事件在训练数据中未出现，但在测试数据中出现时，会导致零概率问题。这会影响模型的性能，因为模型无法正确地对这些事件进行预测，为了解决这个问题，代码中使用了 Laplace 平滑
在评估模型时，如果启用了 Laplace 平滑`laplace=True`，则会在计算概率时对每个事件的计数值加上一个小的平滑因子。这个平滑因子帮助防止在测试数据中出现的事件在训练数据中未出现时导致概率为零的情况

```python
# 代码在 evaluate_model 函数里的第 126 行
## Laplace Smoothing
if laplace:
    prob += sum(math.log((word_freq[y].get(word, 0) + l_index) / (total_freq[y] + len(word_freq[y]) * l_index)) for word in words)
else:
    prob += sum(math.log(word_freq[y].get(word, 1) / total_freq[y]) for word in words)
```

```python
# 没有使用 Laplace 平滑 evaluate_model(laplace=False)
Average Accuracy: 0.9849342284124892
Average Precision: 1.0
Average Recall: 0.9849342284124892
Average F1 Score: 0.9924051852571079
```

```python
# 使用 Laplace 平滑 evaluate_model(laplace=True)
Average Accuracy: 0.9920697659828093
Average Precision: 1.0
Average Recall: 0.9920697659828093
Average F1 Score: 0.9960185666568397
```
从上述数据可以观察到，在启用 Laplace 平滑的情况下，模型的性能略有提升


### **问题3: 特征设计**
`Received: From` 字段是指邮件传输过程中的一个关键字段，记录了邮件从发送者到接收者之间的传输路径，包含了一系列的邮件服务器或者邮件客户端的信息，通过解析和提取 "Received: From" 字段，我们可以设计出一些潜在的特征：

- 发件人 IP 地址的数量：统计邮件经过的不同 IP 地址的数量，这可以反映出邮件的传输路径的复杂程度。
- 发件人 IP 地址的种类：统计邮件经过的不同 IP 地址的种类，可以反映出邮件传输路径中的不同邮件服务器的数量。
- 传输协议的种类：统计邮件传输过程中使用的不同协议的种类，这可以反映出邮件传输的多样性。
- 时间戳的差值：计算邮件经过不同节点之间的时间戳差值，可以反映出邮件在传输路径上的时间消耗。

通过利用 "Received: From" 字段中的信息，我们可以设计出一些能够反映邮件传输过程特征的特征，从而帮助模型更好地理解和分类电子邮件

```python
# 代码在 train 函数的第57行和 evaluate_model 函数的第120行
received_from = re.findall(r'Received: from\s+(\S+)', text)
if received_from:
    next_word = received_from[0].split()[0]
    words.append(next_word)
```

```python
# 没有使用 Received: from 特征设计
Average Accuracy: 0.9849342284124892
Average Precision: 1.0
Average Recall: 0.9849342284124892
Average F1 Score: 0.9924051852571079
```

```python
# 使用 Received: from 特征设计
Average Accuracy: 0.9851987786770395
Average Precision: 1.0
Average Recall: 0.9851987786770395
Average F1 Score: 0.9925387824862023
```

从上述数据可以观察到，使用了特征设计后得到的结果更准确，可以提供一些额外的线索，有助于模型更准确地分类邮件，然而，即使提升幅度较小，使用额外的特征设计仍然是一种有益的实践，可以提高模型的鲁棒性和泛化能力