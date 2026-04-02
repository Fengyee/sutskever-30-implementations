# Day 3: RNN正则化

> 对应论文/Notebook: Zaremba et al. (2014) "Recurrent Neural Network Regularization" / `04_rnn_regularization.ipynb`
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 正则化的必要性

RNN/LSTM 模型通常有大量参数（Day 2 中我们计算过 LSTM 的参数量是标准 RNN 的 4 倍），容易在训练集上过拟合。然而，直接将标准 Dropout 应用于 RNN 的循环连接会导致严重问题——信息在时间步之间传递时被随机丢弃，相当于给模型注入了大量噪声，反而损害了模型学习长期依赖的能力。

### 问题的本质

标准 Dropout 在每个时间步独立采样一个新的二进制 mask，这意味着：
- 时间步 $t$ 被保留的神经元可能在时间步 $t+1$ 被丢弃
- 隐藏状态中的信息被不可预测地破坏
- 随着序列变长，累积的噪声使得长期依赖无法学习

这就引出了本文的核心问题：**如何在不破坏时间依赖性的前提下，有效正则化 RNN？**

---

## 关键公式

### 标准 Dropout 回顾

标准 Dropout (Srivastava et al. 2014) 在训练时以概率 $p$ 随机将激活值置零：

$$\tilde{h} = m \odot h, \quad m_i \sim \text{Bernoulli}(1 - p)$$

测试时乘以 $(1-p)$ 进行缩放，或训练时使用 inverted dropout 除以 $(1-p)$：

$$\tilde{h} = \frac{1}{1-p} \cdot m \odot h$$

### Zaremba 方法：仅对非循环连接应用 Dropout

Zaremba et al. (2014) 的核心发现非常简洁：

**只在 RNN 的非循环连接（即层间连接和输入/输出连接）上应用标准 Dropout，不在循环连接（时间步之间）上应用 Dropout。**

对于多层 LSTM，在第 $l$ 层的时间步 $t$：

$$h_t^l = \text{LSTM}(h_t^{l-1}, h_{t-1}^l, c_{t-1}^l)$$

Dropout 只应用在层间传递的 $h_t^{l-1}$ 上：

$$h_t^l = \text{LSTM}(\text{Dropout}(h_t^{l-1}), h_{t-1}^l, c_{t-1}^l)$$

而 $h_{t-1}^l$（同一层前一时间步的隐藏状态）不经过 Dropout。

### Variational Dropout (Gal & Ghahramani, 2016)

Variational Dropout 是更理论化的改进方案。核心思想：**在整个序列的所有时间步中使用同一个 Dropout mask**。

对于序列中的所有时间步 $t = 1, 2, \ldots, T$：

$$m_x \sim \text{Bernoulli}(1-p_x) \quad \text{（输入 mask，采样一次）}$$
$$m_h \sim \text{Bernoulli}(1-p_h) \quad \text{（隐藏状态 mask，采样一次）}$$

然后在每个时间步：
$$h_t = \text{LSTM}(m_x \odot x_t, \; m_h \odot h_{t-1}, \; c_{t-1})$$

关键区别在于 $m_x$ 和 $m_h$ 在整个序列中**不变**。

**理论依据**：从变分推断的角度，每个时间步使用相同的 mask 等价于对权重矩阵进行近似贝叶斯推断。不同时间步使用不同 mask 则没有这个理论保证。

### Weight Tying（权重绑定）

Weight Tying (Inan et al. 2017; Press & Wolf 2017) 是一种参数共享形式的正则化：

在语言模型中，输入嵌入矩阵 $E \in \mathbb{R}^{|V| \times d}$ 和输出投影矩阵 $W_{out} \in \mathbb{R}^{|V| \times d}$ 共享参数：

$$W_{out} = E$$

即：
$$p(w_t | h_t) = \text{softmax}(E \cdot h_t)$$

**为什么有效？**
- 减少了约 $|V| \times d$ 个参数（词汇表大时非常可观）
- 输入和输出空间被强制对齐：语义相近的词在嵌入空间中也接近
- 等价于一种隐式正则化，防止输出层过拟合

---

## 直觉理解

### Dropout 在 RNN 中的特殊性

想象一个人在听一段很长的故事：
- **标准 Dropout 应用于循环连接**：相当于在故事的每一句话之间，随机让这个人忘掉一部分已知信息。故事越长，丢失的信息越多，最终完全无法理解故事。
- **Zaremba 方法**：只在"做笔记"时随机省略一些内容（层间传递），但"记忆"本身不受干扰（循环连接不 Dropout）。
- **Variational Dropout**：决定好要关注故事的哪些方面（一次性采样 mask），然后在整个故事中一直关注这些方面。

### 为什么共享 mask 更好？

每个时间步使用不同的 mask 会引入 $O(T)$ 的累积噪声。而共享 mask 的噪声是 $O(1)$ 的——无论序列多长，噪声量不变。

数学上，经过 $T$ 步后的方差：
- 独立 mask：$\text{Var} \propto T \cdot p(1-p)$
- 共享 mask：$\text{Var} \propto p(1-p)$（不随 $T$ 增长）

### Dropout 率的经验选择

| 位置 | 推荐 Dropout 率 | 说明 |
|------|-----------------|------|
| 输入层 | 0.2 - 0.4 | 嵌入层需要保留较多信息 |
| 隐藏层之间 | 0.3 - 0.5 | 层间传递的标准正则化 |
| 循环连接 (variational) | 0.2 - 0.3 | 较低的率以保护时间依赖 |
| 输出层 | 0.3 - 0.5 | 防止输出层过拟合 |

---

## 其他 RNN 正则化技术

### Zoneout (Krueger et al. 2017)

Zoneout 是 Dropout 的一个优雅变体，专为 RNN 设计：

不是将激活值置零，而是**随机保留上一时间步的值**：

$$h_t = m_t \odot h_{t-1} + (1 - m_t) \odot \tilde{h}_t$$

其中 $m_t \sim \text{Bernoulli}(p)$，$\tilde{h}_t$ 是当前时间步的正常计算结果。

Zoneout 的好处：
- 被"zoneout"的单元保持不变而非被置零，信息得以保留
- 创造了类似残差连接的效果
- 对 cell state 和隐藏状态可以分别设置不同的 zoneout 率

### AWD-LSTM (Merity et al. 2018)

AWD-LSTM (ASGD Weight-Dropped LSTM) 是多种正则化技术的集大成者：

1. **Weight Dropout (DropConnect)**：对权重矩阵本身应用 Dropout，而非激活值
   $$\tilde{W}_{hh} = m \odot W_{hh}$$

2. **Variational Dropout**：在隐藏状态和输入上使用共享 mask

3. **嵌入 Dropout**：随机将整个词的嵌入置零（而非嵌入向量的单个维度）

4. **权重绑定 (Weight Tying)**

5. **AR/TAR 正则化**：
   - Activation Regularization (AR)：$L_{AR} = \alpha \|h_t\|_2^2$
   - Temporal Activation Regularization (TAR)：$L_{TAR} = \beta \|h_t - h_{t-1}\|_2^2$

   TAR 鼓励隐藏状态在相邻时间步之间平滑变化。

6. **NT-ASGD 优化器**：非单调触发的平均随机梯度下降

### L2 正则化 (Weight Decay)

最基本的正则化方法，在损失函数中加入权重范数惩罚：

$$L_{total} = L_{task} + \lambda \|W\|_2^2$$

对 RNN 同样适用，但效果通常不如专门设计的方法。

---

## 历史背景

### RNN 正则化的发展

- **2012**：Hinton et al. 提出 Dropout，但主要用于前馈网络和 CNN
- **2013-2014**：研究者发现标准 Dropout 直接用于 RNN 循环连接效果很差
- **2014**：Zaremba et al. 提出仅在非循环连接上使用 Dropout 的方案
- **2016**：Gal & Ghahramani 从贝叶斯角度提出 Variational Dropout
- **2017**：Krueger et al. 提出 Zoneout；Press & Wolf, Inan et al. 独立提出 Weight Tying
- **2018**：Merity et al. 将多种技术组合为 AWD-LSTM，在语言建模上创下当时的最佳记录

### Zaremba 论文的意义

Zaremba 等人的论文（共同作者包括 Ilya Sutskever 和 Oriol Vinyals）虽然方法简单，但有几个重要贡献：
1. 首次系统性地证明 Dropout 可以有效正则化 LSTM
2. 明确指出循环连接上不应使用标准 Dropout
3. 在 Penn Treebank 语言建模基准上取得了当时最好的结果
4. 这种简洁的方法至今仍被广泛使用

---

## 与其他论文的关联

| 论文/主题 | 关联 |
|-----------|------|
| **RNN 基础 (Day 1)** | 正则化解决 RNN 的过拟合问题 |
| **LSTM (Day 2)** | LSTM 的 cell state 需要特殊的 Dropout 策略 |
| **网络剪枝 (Day 4)** | 剪枝是另一种形式的正则化——通过减少参数来防止过拟合 |
| **ResNet (Day 6)** | Zoneout 的保留机制与残差连接有理论联系 |
| **Batch Normalization (后续)** | 另一类隐式正则化技术 |
| **Transformer (后续)** | Transformer 中的 Dropout 应用位置与 RNN 不同 |

### 正则化方法的统一视角

所有正则化方法本质上都在做同一件事——**限制模型复杂度**：

| 方法 | 如何限制复杂度 |
|------|---------------|
| L2 正则化 | 惩罚大的权重值 |
| Dropout | 强制网络学习冗余表示 |
| Weight Tying | 减少独立参数数量 |
| 剪枝 | 直接移除参数 |
| 早停 (Early Stopping) | 限制优化步数 |

从 MDL (Minimum Description Length) 的角度（Day 4 内容），这些方法都在减少模型的描述长度。

---

## 检查点问题

完成本节学习后，尝试回答以下问题：

1. **基础理解**：为什么在 RNN 的循环连接上使用标准 Dropout 会破坏长期依赖的学习？用数学或直觉解释。

2. **方法比较**：Zaremba 方法和 Variational Dropout 的核心区别是什么？Variational Dropout 的理论优势体现在哪里？

3. **Weight Tying**：在一个词汇表大小为 50,000、嵌入维度为 512 的语言模型中，Weight Tying 节省了多少参数？这占模型总参数量的比例大约是多少？

4. **实践选择**：如果你要训练一个 LSTM 语言模型，你会选择哪些正则化技术的组合？为什么？

5. **批判性思考**：AWD-LSTM 组合了 6 种以上的正则化技术。这种"技术堆叠"的方法论有什么潜在问题？与 Transformer 时代的简洁架构相比，你怎么看？

---

> **下一步**: Day 4 将学习网络剪枝与 MDL 原则，从信息论的角度理解为什么"压缩=泛化"。
