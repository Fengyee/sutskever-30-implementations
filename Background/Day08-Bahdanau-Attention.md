# Day 8: 注意力机制 (Bahdanau Attention)

> 对应论文/Notebook: Bahdanau et al. 2015 "Neural Machine Translation by Jointly Learning to Align and Translate", notebook: 14_bahdanau_attention.ipynb
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 1. Seq2Seq 的信息瓶颈问题

传统的序列到序列 (Seq2Seq) 模型由编码器 (Encoder) 和解码器 (Decoder) 两部分组成。编码器将整个输入序列压缩为一个**固定长度的上下文向量** $c$，解码器基于该向量生成输出序列。

这一架构存在根本性的缺陷：

- **信息瓶颈**：无论输入序列多长，所有信息都必须压缩到一个固定维度的向量中
- **长序列退化**：当输入序列较长时，编码器难以将所有关键信息有效编码
- **实验证据**：Cho et al. (2014) 的实验表明，传统 Seq2Seq 模型在句子长度超过 20 个词时，翻译质量急剧下降

直觉类比：这就像让一个翻译者先完整听完一段 10 分钟的演讲，然后仅凭记忆翻译全部内容——信息损失是不可避免的。

### 2. 注意力机制的核心思想

Bahdanau 等人提出的注意力机制，核心思想是：**解码器在生成每个输出词时，不再依赖单一固定的上下文向量，而是动态地"关注"输入序列的不同部分**。

这一机制包含三个关键步骤：

1. **对齐分数计算**：评估解码器当前状态与编码器每个隐状态的相关性
2. **注意力权重归一化**：通过 softmax 将对齐分数转化为概率分布
3. **上下文向量加权求和**：根据注意力权重对编码器隐状态进行加权组合

### 3. 对齐模型 (Alignment Model)

对齐模型是注意力机制的核心组件，负责计算解码器与编码器之间的"对齐程度"。Bahdanau 使用了一个前馈神经网络作为对齐函数：

$$e_{ij} = a(s_{i-1}, h_j)$$

其中：
- $s_{i-1}$：解码器在时间步 $i-1$ 的隐状态（即生成第 $i$ 个输出词之前的状态）
- $h_j$：编码器在位置 $j$ 的隐状态
- $a(\cdot)$：对齐函数（一个可学习的前馈神经网络）
- $e_{ij}$：标量对齐分数，表示输入位置 $j$ 对输出位置 $i$ 的重要程度

### 4. 编码器：双向 RNN

Bahdanau 注意力使用**双向 RNN** (Bidirectional RNN) 作为编码器：

$$\overrightarrow{h_j} = \text{GRU}(\overrightarrow{h_{j-1}}, x_j)$$

$$\overleftarrow{h_j} = \text{GRU}(\overleftarrow{h_{j+1}}, x_j)$$

$$h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]$$

每个位置 $j$ 的隐状态 $h_j$ 是前向和后向隐状态的拼接，因此包含了该位置周围的完整上下文信息。

---

## 关键公式

### 对齐分数 (Alignment Score)

$$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

其中 $v_a$、$W_a$、$U_a$ 是可学习参数。这被称为**加性注意力** (Additive Attention)，因为 $s_{i-1}$ 和 $h_j$ 的变换结果通过加法组合。

### 注意力权重 (Attention Weights)

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

即对所有输入位置的对齐分数做 softmax 归一化，使得 $\sum_j \alpha_{ij} = 1$。

### 上下文向量 (Context Vector)

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

上下文向量是编码器所有隐状态的加权和，权重就是注意力权重。每个解码时间步 $i$ 都会计算一个不同的上下文向量 $c_i$。

### 解码器更新

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

解码器的隐状态更新同时考虑：前一时间步的隐状态 $s_{i-1}$、前一时间步的输出 $y_{i-1}$、以及当前的动态上下文向量 $c_i$。

---

## 直觉理解

### 注意力即"软对齐"

在传统机器翻译中，对齐 (alignment) 是一个离散的概念——源语言的每个词对应目标语言的某个（或某些）词。Bahdanau 注意力将这一离散对齐"软化"为连续的权重分布。

例如，翻译法语 "L'accord sur l'Espace economique europeen" 为英语时：

- 生成 "European" 时，注意力主要集中在 "europeen" 上
- 生成 "Economic" 时，注意力主要集中在 "economique" 上
- 生成 "Area" 时，注意力主要集中在 "Espace" 上

这种软对齐可以处理语言之间的词序差异（如形容词位置）。

### 为什么选择加性注意力？

Bahdanau 选择了加性 (additive) 形式而非点积 (dot-product) 形式，因为：

1. 解码器隐状态和编码器隐状态的维度可能不同
2. 加性形式通过独立的线性变换 $W_a$ 和 $U_a$ 将两者映射到相同的空间
3. 非线性激活 $\tanh$ 增加了模型的表达能力

### 加性注意力 vs 乘性注意力

| 特性 | 加性注意力 (Bahdanau) | 乘性注意力 (Luong) |
|------|----------------------|-------------------|
| 公式 | $e_{ij} = v_a^T \tanh(W_a s_i + U_a h_j)$ | $e_{ij} = s_i^T W h_j$ 或 $s_i^T h_j$ |
| 计算复杂度 | 较高（额外参数和非线性） | 较低（矩阵乘法可高度并行化） |
| 低维表现 | 较好 | 较好 |
| 高维表现 | 较好 | 需要缩放（导致了 Transformer 的 $\sqrt{d_k}$） |
| 提出时间 | Bahdanau et al. 2015 | Luong et al. 2015 |

Luong 在同年提出的乘性注意力 (Multiplicative Attention) 更为简洁：

$$e_{ij} = s_i^T h_j \quad \text{(dot-product)}$$

$$e_{ij} = s_i^T W h_j \quad \text{(general)}$$

两种形式在实践中表现相当，但乘性注意力在计算上更高效，为后续 Transformer 的设计铺平了道路。

### 注意力的信息论解读

从信息论角度看，注意力机制实质上是一种**信息路由**机制：

- 传统 Seq2Seq：所有信息必须通过一个固定容量的"管道"（瓶颈向量）
- 注意力机制：为每个输出时间步建立一个**动态带宽**的信息通道，按需从输入的不同位置提取信息

---

## 历史背景

### 时间线

- **2014 年 9 月**：Sutskever et al. 发表 "Sequence to Sequence Learning with Neural Networks"，奠定了 Seq2Seq 架构基础
- **2014 年 9 月**：Bahdanau et al. 在 arXiv 提交注意力机制论文（ICLR 2015 发表）
- **2015 年 8 月**：Luong et al. 提出乘性注意力变体
- **2017 年 6 月**：Vaswani et al. 发表 "Attention Is All You Need"，将注意力机制发展为 Transformer 的核心

### 研究动机

Bahdanau 的工作直接受到以下两个观察的驱动：

1. **经验观察**：Cho et al. (2014) 发现 Seq2Seq 模型在长句上表现显著下降
2. **认知启发**：人类翻译时并不是先记住整个句子再翻译，而是在翻译过程中反复回顾原文的不同部分

### 影响深远

Bahdanau 注意力的影响远超机器翻译：

- **计算机视觉**：Show, Attend and Tell (2015) 将注意力用于图像描述
- **语音识别**：Listen, Attend and Spell (2016) 将注意力用于端到端语音识别
- **Transformer**：Self-Attention 是注意力机制的自然延伸，成为现代深度学习的基石

---

## 与其他论文的关联

### 与 Seq2Seq (Day 1-3) 的关系

Bahdanau 注意力是对标准 Seq2Seq 架构的**直接改进**。标准 Seq2Seq 可以看作注意力机制的特例——所有注意力权重集中在编码器最后一个时间步上。

### 与 Pointer Networks (Day 9) 的关系

Pointer Networks 将注意力权重直接作为输出（"指向"输入位置），而 Bahdanau 注意力将加权上下文向量作为辅助输入。Pointer Networks 可以看作注意力机制的一种极端应用。

### 与 Transformer (Day 10) 的关系

Transformer 中的缩放点积注意力 (Scaled Dot-Product Attention) 是 Bahdanau 注意力的演化版本：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

从 Bahdanau 到 Transformer，核心变化是：
1. 从加性注意力到缩放点积注意力
2. 从 RNN 编码器的隐状态到 Query/Key/Value 的显式分离
3. 从单头到多头注意力
4. 从编码器-解码器注意力到自注意力

### 与 GNN (Day 11) 的关系

图注意力网络 (GAT) 将注意力机制应用于图结构，可以看作 Bahdanau 注意力在非序列数据上的推广。

---

## 检查点问题

1. **传统 Seq2Seq 的信息瓶颈问题是什么？为什么注意力机制能解决它？**
   > 提示：考虑固定长度向量能承载的信息量与序列长度的关系。

2. **写出 Bahdanau 注意力的三个核心公式（对齐分数、注意力权重、上下文向量）。**
   > 提示：$e_{ij}$、$\alpha_{ij}$、$c_i$ 分别是什么？

3. **加性注意力和乘性注意力的主要区别是什么？各自的优缺点？**
   > 提示：考虑计算复杂度和表达能力。

4. **为什么 Bahdanau 使用双向 RNN 作为编码器？如果使用单向 RNN 会有什么问题？**
   > 提示：考虑每个位置的隐状态包含的上下文信息范围。

5. **注意力权重矩阵可以可视化为什么？它在机器翻译任务中揭示了什么？**
   > 提示：想想源语言和目标语言之间的词对应关系。

6. **如果把注意力权重全部设为均匀分布（即 $\alpha_{ij} = 1/T_x$），模型会退化成什么？**
   > 提示：此时上下文向量 $c_i$ 等于什么？

---

> **下一步**：完成理论阅读后，请打开 `14_bahdanau_attention.ipynb` 进行代码实践。
