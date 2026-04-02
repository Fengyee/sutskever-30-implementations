# Day 11: 图神经网络 (Graph Neural Networks)

> 对应论文/Notebook: Kipf & Welling 2017 "Semi-Supervised Classification with Graph Convolutional Networks" 等, notebook: 12_graph_neural_networks.ipynb
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 1. 为什么需要图神经网络？

传统深度学习模型处理的是**规则结构**的数据：

- **CNN**：网格结构（图像的像素网格）
- **RNN/Transformer**：序列结构（文本、语音）

然而，现实世界中大量数据天然呈现**图结构**：

- **社交网络**：用户为节点，关注/好友关系为边
- **分子结构**：原子为节点，化学键为边
- **引用网络**：论文为节点，引用关系为边
- **知识图谱**：实体为节点，关系为边
- **交通网络**：路口为节点，道路为边

图神经网络 (GNN) 的核心目标是在**非规则图结构**上进行表示学习。

### 2. 图的基本定义

一个图 $G = (V, E)$ 由以下部分组成：

- **节点集合** $V = \{v_1, v_2, \dots, v_n\}$
- **边集合** $E \subseteq V \times V$
- **邻接矩阵** $A \in \{0, 1\}^{n \times n}$，其中 $A_{ij} = 1$ 表示节点 $i$ 和 $j$ 之间存在边
- **节点特征矩阵** $X \in \mathbb{R}^{n \times d}$，每一行是一个节点的 $d$ 维特征向量
- **度矩阵** $D$，对角矩阵，$D_{ii} = \sum_j A_{ij}$

### 3. 消息传递框架 (Message Passing Framework)

几乎所有 GNN 都可以统一在消息传递框架下理解。每一层 GNN 的操作包含两步：

**第一步：聚合 (Aggregate)**

每个节点收集其邻居节点的信息：

$$m_v^{(l)} = \text{AGGREGATE}^{(l)}\left(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)$$

其中 $\mathcal{N}(v)$ 是节点 $v$ 的邻居集合，$h_u^{(l-1)}$ 是节点 $u$ 在第 $l-1$ 层的特征表示。

**第二步：更新 (Update)**

每个节点结合自身信息和聚合的邻居信息来更新表示：

$$h_v^{(l)} = \text{UPDATE}^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)$$

不同的 GNN 变体主要在 AGGREGATE 和 UPDATE 函数的选择上有所不同。

### 4. GCN (Graph Convolutional Network)

Kipf & Welling (2017) 提出的 GCN 是最经典的图神经网络之一。其层间传播规则为：

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

其中：
- $\tilde{A} = A + I_n$：加入自连接的邻接矩阵
- $\tilde{D}$：$\tilde{A}$ 对应的度矩阵，$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- $H^{(l)} \in \mathbb{R}^{n \times d_l}$：第 $l$ 层所有节点的特征矩阵
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$：可学习的权重矩阵
- $\sigma$：非线性激活函数（如 ReLU）

**逐步理解这个公式**：

1. $H^{(l)} W^{(l)}$：对所有节点特征进行线性变换
2. $\tilde{A} \cdot (\cdot)$：每个节点聚合自身和邻居的变换后特征
3. $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$：对称归一化，防止度数大的节点主导聚合

### 5. GraphSAGE：采样 + 聚合

Hamilton et al. (2017) 提出的 GraphSAGE 解决了 GCN 的两个局限：

- **GCN 局限 1**：需要在整个图上进行全局计算（不适用于大图）
- **GCN 局限 2**：无法处理训练时未见过的新节点（归纳学习）

GraphSAGE 的核心思想：

1. **采样邻居**：每个节点随机采样固定数量的邻居（而非使用所有邻居）
2. **聚合信息**：使用可学习的聚合函数
3. **拼接更新**：将聚合结果与自身特征拼接后变换

$$h_{\mathcal{N}(v)}^{(l)} = \text{AGGREGATE}^{(l)}\left(\{h_u^{(l-1)} : u \in \mathcal{S}(v)\}\right)$$

$$h_v^{(l)} = \sigma\left(W^{(l)} \cdot [h_v^{(l-1)} \| h_{\mathcal{N}(v)}^{(l)}]\right)$$

其中 $\mathcal{S}(v)$ 是采样的邻居子集，$[\cdot \| \cdot]$ 表示拼接。

聚合函数可以是：

- **Mean**：$\text{AGGREGATE} = \text{mean}(\{h_u\})$
- **LSTM**：将邻居随机排列后用 LSTM 处理
- **Pooling**：$\text{AGGREGATE} = \max(\{\sigma(W_{\text{pool}} h_u + b)\})$

### 6. 邻接矩阵在消息传递中的作用

邻接矩阵 $A$ 是 GNN 的核心数据结构。矩阵乘法 $AH$ 的本质是**邻居特征的聚合**：

$$(AH)_i = \sum_{j} A_{ij} H_j = \sum_{j \in \mathcal{N}(i)} H_j$$

即第 $i$ 行的结果是节点 $i$ 的所有邻居特征的求和。

**加入自连接** $\tilde{A} = A + I$：让每个节点在聚合时也包含自身信息。

**对称归一化** $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$：
- 如果不归一化，度数高的节点会获得更大的聚合值
- 对称归一化使得每条边的贡献被两端节点的度数调节
- 对应的每个元素：$(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{D}_{ii}} \sqrt{\tilde{D}_{jj}}}$

### 7. 过平滑 (Over-Smoothing) 问题

随着 GNN 层数的增加，一个严重的问题出现了：**所有节点的表示趋于相同**。

**原因分析**：

- 每一层 GNN，每个节点聚合其 1-hop 邻居的信息
- $k$ 层 GNN 后，每个节点的感受野扩展到 $k$-hop 邻居
- 当 $k$ 足够大时，大多数节点的 $k$-hop 邻居覆盖了整个图
- 所有节点看到相似的全局信息，表示趋于收敛

**数学分析**：

对于归一化的邻接矩阵 $\hat{A}$，反复乘以 $\hat{A}$ 相当于图上的**随机游走**。当步数趋向无穷时，随机游走收敛到**平稳分布**——这就是过平滑的数学本质。

**解决方案**：

- **限制层数**：通常 2-3 层 GNN 即可（与 CNN 动辄几十层形成鲜明对比）
- **残差连接**：$h_v^{(l)} = h_v^{(l)} + h_v^{(l-1)}$
- **跳跃连接**：将各层的输出拼接作为最终表示
- **DropEdge**：随机丢弃部分边，减缓信息传播速度

---

## 关键公式

### GCN 传播规则

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

### 单节点视角

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} h_u^{(l)} W^{(l)}\right)$$

其中 $\tilde{d}_v = |\mathcal{N}(v)| + 1$（加上自连接的度）。

### 节点分类的完整流程

1. 输入特征：$H^{(0)} = X \in \mathbb{R}^{n \times d}$
2. 第一层：$H^{(1)} = \text{ReLU}\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W^{(0)}\right)$
3. 第二层：$Z = \text{softmax}\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(1)} W^{(1)}\right)$
4. 损失函数：$\mathcal{L} = -\sum_{v \in V_L} \sum_{c=1}^{C} Y_{vc} \ln Z_{vc}$

其中 $V_L$ 是有标签的节点集合（半监督学习场景）。

### 图级别表示

若需要得到整个图的表示（如图分类任务），需要**图级别的池化**：

$$h_G = \text{READOUT}(\{h_v^{(L)} : v \in V\})$$

常用的 READOUT 函数包括：求和 (sum)、均值 (mean)、最大值 (max)。

---

## 直觉理解

### GNN 与 CNN 的类比

GNN 可以看作 CNN 在不规则结构上的推广：

- **CNN 卷积核**：在固定的网格邻域上滑动，聚合邻域像素
- **GNN 消息传递**：在图的邻居节点上聚合，邻域结构由图决定

关键区别：CNN 的邻域是固定的（如 3x3），而 GNN 每个节点的邻居数量可能不同。

### 频域视角

从谱图理论的角度，GCN 的归一化邻接矩阵操作可以理解为**低通滤波**——它平滑了图信号，使相邻节点的表示更为接近。这解释了为什么 GCN 在"同质性"图（相连节点倾向于有相同标签）上表现良好。

### 为什么 GNN 通常很浅？

在 CNN 中，深度是关键——深层网络可以学习层次化的特征。但在 GNN 中，通常 2-3 层就足够了。原因在于：

- 社交网络中的"六度分隔"理论：大多数节点在 6 步之内可达
- 2-3 层 GNN 的感受野已经覆盖了大量邻居
- 更多层数反而会导致过平滑

---

## 历史背景

### 时间线

- **2005**：Gori et al. 提出 Graph Neural Network 的概念
- **2009**：Scarselli et al. 发表 "The Graph Neural Network Model"
- **2014**：Bruna et al. 提出谱域 (spectral) 图卷积
- **2016**：Defferrard et al. 提出 ChebNet，用切比雪夫多项式近似谱卷积
- **2017**：Kipf & Welling 提出 GCN，简化 ChebNet 为一阶近似
- **2017**：Hamilton et al. 提出 GraphSAGE，支持归纳学习
- **2018**：Velickovic et al. 提出 GAT (Graph Attention Network)，引入注意力权重
- **2019**：Xu et al. 提出 GIN (Graph Isomorphism Network)，分析 GNN 的表达能力
- **2020**：Dwivedi et al. 提出 Graph Transformer

### GCN 的推导来源

Kipf & Welling 的 GCN 公式并非凭空而来。它来自谱图卷积的一阶近似：

1. 谱图卷积：$g_\theta \star x = U g_\theta(\Lambda) U^T x$（需要计算图的特征分解，$O(n^3)$）
2. 切比雪夫近似：用 $K$ 阶切比雪夫多项式近似 $g_\theta(\Lambda)$
3. GCN：取 $K = 1$，进一步简化，得到最终的传播公式

### GNN 的爆发式增长

GCN 论文 (Kipf & Welling, 2017) 是近年来引用最多的深度学习论文之一，催生了 GNN 领域的爆发式增长。主要应用领域包括：

- 药物发现（分子属性预测）
- 推荐系统（用户-物品交互图）
- 社交网络分析
- 交通流量预测
- 物理模拟

---

## 与其他论文的关联

### 与 Transformer (Day 10) 的关系

Self-Attention 和 GNN 的消息传递有深刻联系：

- **Self-Attention = 全连接图上的 GNN**：在 Transformer 中，每个 token 与所有其他 token 交互，等价于在一个完全图上进行消息传递
- **GNN = 稀疏图上的 Attention**：GNN 只在有边相连的节点之间进行消息传递

Graph Attention Network (GAT) 明确将注意力机制引入 GNN：

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T [Wh_i \| Wh_k]))}$$

### 与 Bahdanau 注意力 (Day 8) 的关系

GAT 中的注意力机制与 Bahdanau 的加性注意力形式非常相似。区别在于 GAT 只在图的邻居之间计算注意力，而 Bahdanau 在编码器-解码器之间计算。

### 与 Seq2Seq for Sets (Day 12) 的关系

GNN 的一个子问题是**图级别表示**：如何将一个图（即节点的集合 + 边）表示为一个向量。这与集合编码问题 (Seq2Seq for Sets) 密切相关——图级别的 READOUT 函数本质上是一种集合池化操作。

### 与 ResNet (Day 13) 的关系

GNN 中的残差连接直接借鉴了 ResNet 的设计，用于缓解过平滑问题（类似于 ResNet 解决的梯度消失问题）。

---

## 检查点问题

1. **消息传递框架的两个基本步骤是什么？不同 GNN 变体主要在哪一步上有区别？**
   > 提示：Aggregate 和 Update。

2. **GCN 公式中 $\tilde{A} = A + I$ 的意义是什么？如果不加自连接会怎样？**
   > 提示：考虑节点自身信息在聚合中的保留。

3. **对称归一化 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 的目的是什么？它对高度数节点有什么影响？**
   > 提示：不归一化时，度数高的节点会怎样？

4. **什么是过平滑问题？为什么 GNN 通常比 CNN 浅很多？**
   > 提示：考虑感受野的扩展速度。

5. **Self-Attention 和 GNN 消息传递有什么联系？Transformer 可以看作什么图上的 GNN？**
   > 提示：完全图上的消息传递。

6. **GraphSAGE 相比 GCN 的主要改进是什么？它为什么能处理新节点？**
   > 提示：归纳学习 vs 转导学习。

---

> **下一步**：完成理论阅读后，请打开 `12_graph_neural_networks.ipynb` 进行代码实践。
