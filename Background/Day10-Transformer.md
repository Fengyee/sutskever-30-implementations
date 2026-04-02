# Day 10: Transformer: Self-Attention

> 对应论文/Notebook: Vaswani et al. 2017 "Attention Is All You Need", notebook: 13_attention_is_all_you_need.ipynb
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 1. 从 RNN 到 Self-Attention：动机

RNN 系列模型（LSTM、GRU）在序列建模中存在两个根本性问题：

- **顺序计算瓶颈**：RNN 必须按时间步顺序处理序列，无法充分利用现代 GPU 的并行计算能力
- **长距离依赖衰减**：即使有门控机制，信息在长序列中传递时仍会衰减

Transformer 的核心创新是：**完全抛弃循环结构，仅使用注意力机制来捕获序列中的依赖关系**。

### 2. 自注意力 (Self-Attention) 的基本原理

自注意力的核心思想是：序列中的**每个位置都与所有其他位置进行交互**，直接计算任意两个位置之间的关系。

与 Bahdanau 注意力的区别：

| 特性 | Bahdanau 注意力 | Self-Attention |
|------|----------------|---------------|
| Query 来源 | 解码器隐状态 | 同一序列的当前位置 |
| Key/Value 来源 | 编码器隐状态 | 同一序列的所有位置 |
| 作用 | 编码器-解码器之间 | **序列内部** |

### 3. Query, Key, Value 的概念

自注意力将每个输入位置的表示映射为三个向量：

- **Query (Q)**：当前位置"想要查找什么信息"
- **Key (K)**：每个位置"提供什么信息的索引"
- **Value (V)**：每个位置"实际包含的信息"

直觉类比：在图书馆中，Query 是你的搜索请求，Key 是书籍的索引标签，Value 是书的实际内容。你用 Query 与所有 Key 匹配，找到最相关的 Key 后，取出对应的 Value。

### 4. 缩放点积注意力 (Scaled Dot-Product Attention)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $d_k$ 是 Key 向量的维度。

### 5. 多头注意力 (Multi-Head Attention)

与其使用单一的注意力函数，Transformer 采用多头注意力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

其中每个头：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

多头注意力允许模型在**不同的表示子空间**中同时关注不同位置的信息。

### 6. 位置编码 (Positional Encoding)

由于 Self-Attention 本身是**排列等变** (permutation equivariant) 的——它不区分输入的顺序——因此需要显式注入位置信息。

Transformer 使用正弦/余弦函数生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 7. Encoder-Decoder 整体架构

**编码器** (N=6 层，每层包含)：
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network
3. 每个子层都有残差连接 (Residual Connection) + 层归一化 (Layer Normalization)

**解码器** (N=6 层，每层包含)：
1. Masked Multi-Head Self-Attention（防止看到未来信息）
2. Multi-Head Cross-Attention（关注编码器输出）
3. Position-wise Feed-Forward Network
4. 每个子层同样有残差连接 + 层归一化

### 8. 前馈网络 (Feed-Forward Network)

每个注意力层后面都跟着一个逐位置的前馈网络：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

即两个线性变换中间夹一个 ReLU 激活。内层维度 $d_{ff} = 2048$，远大于模型维度 $d_{\text{model}} = 512$。

---

## 关键公式

### 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

计算步骤分解：

1. **相似度计算**：$QK^T \in \mathbb{R}^{n \times n}$，得到所有位置对之间的点积
2. **缩放**：除以 $\sqrt{d_k}$，防止点积值过大
3. **归一化**：softmax 将每行转化为概率分布
4. **加权求和**：与 $V$ 相乘得到输出

### 缩放因子 $\sqrt{d_k}$ 的作用

当 $d_k$ 较大时，$Q$ 和 $K$ 的点积的方差约为 $d_k$（假设 $Q$ 和 $K$ 的每个分量独立且均值为 0、方差为 1）：

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

因此 $q \cdot k$ 的标准差为 $\sqrt{d_k}$。当 $d_k = 512$ 时，点积值可能非常大，导致 softmax 函数进入**梯度极小的饱和区域**。除以 $\sqrt{d_k}$ 将方差归一化为 1，确保 softmax 的输入在合理范围内。

### 多头注意力参数

$$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

$$W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$$

论文中 $h = 8$，$d_k = d_v = d_{\text{model}} / h = 64$。

总参数量与单头注意力（使用完整维度）相当，但多头注意力能捕获更丰富的关系。

### 残差连接与层归一化

$$\text{SubLayerOutput} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

残差连接确保梯度可以直接回传到较早的层，层归一化稳定训练过程。

### 位置编码的性质

$$PE_{(pos+k)} \text{ 可以表示为 } PE_{(pos)} \text{ 的线性变换}$$

这意味着模型可以通过学习线性变换来关注相对位置，而不仅是绝对位置。

---

## 直觉理解

### Self-Attention 是什么？

想象一个圆桌讨论：每个人（位置）同时听取所有其他人（位置）的发言，并根据与自己问题（Query）的相关程度（Key 的匹配度）来决定关注谁。最终每个人综合所有人的信息（Value 的加权和）来更新自己的理解。

### 为什么需要多头？

单头注意力就像只从一个角度看问题。多头注意力让模型从多个角度同时观察：

- 某些头可能关注**语法关系**（主语-谓语）
- 某些头可能关注**语义关系**（同义词、反义词）
- 某些头可能关注**位置关系**（相邻词）

研究发现，不同的注意力头确实学到了不同类型的语言关系。

### Masked Attention 的必要性

在解码器中，生成第 $i$ 个词时不应该"看到"第 $i+1$ 及之后的词（否则就是"作弊"）。Masked Attention 通过将注意力矩阵中未来位置的值设为 $-\infty$（softmax 后变为 0）来实现这一约束。

### 位置编码的设计思想

选择 sin/cos 函数的原因：

1. **有界性**：值始终在 $[-1, 1]$ 范围内
2. **唯一性**：每个位置的编码是唯一的
3. **相对位置可学习**：$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，模型可以轻松学习相对位置关系
4. **可外推**：理论上可以处理比训练时更长的序列（尽管实践中效果有限）

### 计算复杂度分析

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|--------|-----------|-----------|------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| RNN | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| CNN (kernel $k$) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

Self-Attention 的关键优势：**任意两个位置之间的路径长度为 $O(1)$**，而 RNN 为 $O(n)$。这意味着长距离依赖的信息传递不再衰减。

代价是 $O(n^2)$ 的计算复杂度，这限制了 Transformer 处理极长序列的能力。

---

## 历史背景

### 时间线

- **2015**：Bahdanau 注意力、Luong 注意力确立
- **2016**：Google Neural Machine Translation (GNMT) 系统，仍基于 RNN + Attention
- **2017 年 6 月**：Vaswani et al. 在 arXiv 提交 "Attention Is All You Need"
- **2017 年 12 月**：NeurIPS 2017 正式发表，成为会议最具影响力的论文之一
- **2018**：BERT (Devlin et al.) 和 GPT (Radford et al.) 证明 Transformer 在预训练中的巨大潜力
- **2020**：Vision Transformer (ViT) 将 Transformer 扩展到计算机视觉
- **2022-2023**：ChatGPT / GPT-4 证明 Transformer 的规模化潜力

### 论文标题的含义

"Attention Is All You Need" 不仅是对模型架构的描述（只用注意力，不用 RNN/CNN），也是对研究范式的宣言。这个标题暗示：之前被认为必不可少的循环结构其实是多余的。

### 作者团队

论文的八位作者来自 Google Brain 和 Google Research，包括：
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- 其中多人后来创办了有影响力的 AI 公司（Cohere、Adept、Essential AI 等）

---

## 与其他论文的关联

### 与 Bahdanau 注意力 (Day 8) 的关系

Transformer 中的注意力是 Bahdanau 注意力的直接演化：

- **对齐函数**：从加性（$v^T \tanh(Ws + Uh)$）变为缩放点积（$QK^T / \sqrt{d_k}$）
- **Q/K/V 分离**：Bahdanau 中 Q 来自解码器、K/V 来自编码器；Transformer 将这一思想推广为通用的 Q/K/V 框架
- **自注意力**：Bahdanau 注意力只在编码器-解码器之间；Transformer 增加了序列内部的自注意力

### 与 Pointer Networks (Day 9) 的关系

Pointer Networks 用注意力权重作为输出。在 Transformer 中，Cross-Attention 也可以看作一种"指向"编码器输出的机制。后来的 Copy Mechanism 变体在 Transformer 中也有应用。

### 与 GNN (Day 11) 的关系

Self-Attention 可以理解为**全连接图上的消息传递**。在一个 $n$ 节点的完全图中，每个节点从所有其他节点接收消息，权重由注意力决定——这正是 Self-Attention 的操作。

### 与 Seq2Seq for Sets (Day 12) 的关系

Transformer 的 Self-Attention 天然具有**排列等变性** (permutation equivariance)：如果改变输入的顺序，输出也相应改变顺序。这与集合处理的需求高度吻合——正是通过位置编码才打破了这种等变性。

### 与 ResNet (Day 13) 的关系

Transformer 大量使用残差连接，这直接来自 ResNet 的设计。残差连接 + 层归一化是 Transformer 深层训练稳定性的关键。

---

## 检查点问题

1. **为什么需要 $\sqrt{d_k}$ 缩放因子？如果去掉它会发生什么？**
   > 提示：考虑 softmax 的输入值范围和梯度行为。

2. **多头注意力相比单头注意力有什么优势？参数量如何对比？**
   > 提示：$h$ 个头，每个头的维度是 $d_k = d_{\text{model}} / h$。

3. **为什么 Transformer 需要位置编码？如果去掉位置编码，模型会丧失什么能力？**
   > 提示：Self-Attention 的排列等变性意味着什么？

4. **解码器中的 Masked Attention 是如何实现的？为什么需要它？**
   > 提示：考虑自回归生成的因果性约束。

5. **Self-Attention 的计算复杂度是 $O(n^2 d)$，这在什么情况下成为瓶颈？有哪些解决方案？**
   > 提示：想想处理长文档或长序列时的情况。

6. **FFN 层的内层维度 (2048) 为什么远大于模型维度 (512)？它起什么作用？**
   > 提示：考虑 FFN 在注意力之后的功能互补性。

---

> **下一步**：完成理论阅读后，请打开 `13_attention_is_all_you_need.ipynb` 进行代码实践。
