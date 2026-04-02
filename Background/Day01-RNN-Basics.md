# Day 1: RNN基础与字符级语言模型

> 对应论文/Notebook: Karpathy's char-rnn / `02_char_rnn_karpathy.ipynb`
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 什么是循环神经网络 (RNN)

循环神经网络是一类专门处理**序列数据**的神经网络。与前馈网络不同，RNN 在每个时间步维护一个**隐藏状态** (hidden state)，使其能够"记住"之前看到的信息。

RNN 的核心思想非常简单：在处理序列中的每个元素时，网络不仅考虑当前输入，还考虑之前所有输入的累积信息（以隐藏状态的形式保存）。

### RNN 的三种基本架构

1. **一对多 (one-to-many)**：单个输入生成序列输出（如图像描述生成）
2. **多对一 (many-to-one)**：序列输入生成单个输出（如情感分析）
3. **多对多 (many-to-many)**：序列输入生成序列输出（如机器翻译、字符级语言模型）

### 字符级语言模型

字符级语言模型是 RNN 最经典的应用之一。模型逐字符读取文本，学习预测下一个字符。训练完成后，可以通过采样生成全新的文本。

工作流程：
1. 将每个字符编码为 one-hot 向量
2. 依次输入 RNN，更新隐藏状态
3. 在每个时间步输出下一个字符的概率分布
4. 用 softmax 将输出转换为概率
5. 训练时用交叉熵损失与真实下一个字符对比

---

## 关键公式

### RNN 前向传播

RNN 的核心计算可以用以下公式描述：

**隐藏状态更新：**

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

其中：
- $h_t \in \mathbb{R}^{d_h}$：时间步 $t$ 的隐藏状态
- $h_{t-1} \in \mathbb{R}^{d_h}$：上一时间步的隐藏状态
- $x_t \in \mathbb{R}^{d_x}$：时间步 $t$ 的输入
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$：隐藏状态到隐藏状态的权重矩阵
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$：输入到隐藏状态的权重矩阵
- $b_h \in \mathbb{R}^{d_h}$：偏置项

**输出计算：**

$$y_t = W_{hy} \cdot h_t + b_y$$

$$p_t = \text{softmax}(y_t)$$

其中 $W_{hy} \in \mathbb{R}^{d_y \times d_h}$ 是隐藏状态到输出的权重矩阵。

**损失函数（交叉熵）：**

$$L_t = -\sum_{k} \hat{y}_{t,k} \log(p_{t,k})$$

总损失为所有时间步损失之和：

$$L = \sum_{t=1}^{T} L_t$$

### BPTT (Backpropagation Through Time) 推导

BPTT 是标准反向传播在时间维度上的展开。关键在于理解梯度如何沿时间步回传。

对于损失 $L$ 关于 $W_{hh}$ 的梯度：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

由链式法则展开每个时间步：

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

其中从 $h_t$ 到 $h_k$ 的梯度传播涉及连乘：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} W_{hh}^T \cdot \text{diag}(1 - h_i^2)$$

这里 $\text{diag}(1 - h_i^2)$ 是 $\tanh$ 的导数的对角矩阵。

### 梯度消失/爆炸的数学解释

连乘项 $\prod_{i=k+1}^{t} W_{hh}^T \cdot \text{diag}(1 - h_i^2)$ 是问题的根源：

**梯度消失：** 当 $\|W_{hh}\|$ 的最大特征值 $< 1$ 时，随着 $t - k$ 增大，连乘项指数级趋近于零。加之 $\tanh$ 的导数 $\in (0, 1]$，进一步加速衰减。

$$\left\| \frac{\partial h_t}{\partial h_k} \right\| \leq \|W_{hh}\|^{t-k} \cdot \gamma^{t-k}$$

其中 $\gamma = \max(1 - h_i^2) \leq 1$。当 $(t-k)$ 足够大时，梯度趋近于零。

**梯度爆炸：** 当 $\|W_{hh}\|$ 的最大特征值 $> 1$ 时，连乘项指数级增长，导致参数更新过大。

**实践中的应对措施：**
- **梯度裁剪 (Gradient Clipping)**：当梯度范数超过阈值时按比例缩放
  $$\text{if } \|g\| > \theta: \quad g \leftarrow \frac{\theta}{\|g\|} \cdot g$$
- **使用 LSTM/GRU** 等门控机制（Day 2 内容）
- **合理初始化** $W_{hh}$（如正交初始化）

---

## 直觉理解

### 为什么用 tanh？

$\tanh$ 的输出范围是 $(-1, 1)$，具有以下优势：
- **零中心化**：不像 sigmoid 那样输出总是正的，避免梯度更新的系统性偏差
- **非线性**：赋予网络拟合复杂函数的能力
- **有界性**：防止隐藏状态的值无限增长

### 字符级 vs 词级语言模型

| 特性 | 字符级 | 词级 |
|------|--------|------|
| 词汇表大小 | ~100 字符 | ~50,000+ 词 |
| 序列长度 | 很长 | 较短 |
| 处理未知词 | 天然支持 | 需要 `<UNK>` 标记 |
| 生成拼写错误 | 可能 | 不会 |
| 计算开销 | 序列长但单步快 | 序列短但 softmax 开销大 |

### 温度采样 (Temperature Sampling)

生成文本时，可以通过温度参数 $\tau$ 控制输出的多样性：

$$p_i = \frac{\exp(y_i / \tau)}{\sum_j \exp(y_j / \tau)}$$

- $\tau \to 0$：趋向于贪心选择（确定性强，但可能重复）
- $\tau = 1$：标准 softmax（平衡多样性与质量）
- $\tau \to \infty$：趋向于均匀分布（随机性最强）

### 隐藏状态的可视化理解

Karpathy 在其博客中展示了 RNN 隐藏状态中某些神经元会自动学到有意义的特征：
- 有的神经元追踪是否在引号内
- 有的神经元追踪缩进层级
- 有的神经元追踪换行位置

这说明 RNN 能够自主发现序列中的结构性模式。

---

## 历史背景

### RNN 的发展时间线

- **1986**：Rumelhart, Hinton & Williams 提出反向传播，为 RNN 训练奠定基础
- **1990**：Elman 提出简单循环网络 (Simple Recurrent Network, SRN)
- **1994**：Bengio 等人首次系统分析 RNN 中的梯度消失问题
- **1997**：Hochreiter & Schmidhuber 发明 LSTM，解决长期依赖问题
- **2013**：Sutskever 等人用深层 LSTM 实现序列到序列学习
- **2015**：Karpathy 发表 "The Unreasonable Effectiveness of Recurrent Neural Networks" 博文，使 char-rnn 广为人知

### Karpathy 的 char-rnn 贡献

Andrej Karpathy 在 2015 年的博文是深度学习普及的里程碑之一。他展示了一个简单的字符级 RNN 可以学会：
- 生成类似莎士比亚的文本
- 生成合法的 LaTeX 代码
- 生成看起来像 C 代码的文本（包括合理的缩进和括号匹配）
- 生成 Linux 内核代码风格的文本

这些惊人的结果让更多人关注到深度学习的潜力。

---

## 与其他论文的关联

| 论文/主题 | 关联 |
|-----------|------|
| **LSTM (Day 2)** | 通过门控机制解决本文讨论的梯度消失问题 |
| **RNN 正则化 (Day 3)** | 防止 RNN 过拟合的技术 |
| **Seq2Seq** | RNN 的编码器-解码器架构，是 NLP 的重要范式 |
| **Attention (后续)** | 替代单一隐藏状态的信息传递方式 |
| **Transformer (后续)** | 最终用自注意力机制替代了 RNN 的循环结构 |

---

## 检查点问题

完成本节学习后，尝试回答以下问题：

1. **基础理解**：RNN 的隐藏状态 $h_t$ 在概念上代表什么？它与前馈网络的中间层有什么本质区别？

2. **公式推导**：写出 BPTT 中 $\frac{\partial L}{\partial W_{hh}}$ 的完整推导，解释为什么需要对所有时间步求和。

3. **梯度问题**：假设 $W_{hh}$ 的最大特征值为 0.9，$\tanh$ 导数上界为 1。经过 100 个时间步后，梯度大约衰减到原来的多少倍？这对学习有什么影响？

4. **实践应用**：在字符级语言模型中，为什么隐藏状态维度的选择很重要？太小或太大会有什么问题？

5. **批判性思考**：RNN 声称可以处理任意长度的序列，但实际上受到什么限制？这些限制是理论上的还是工程上的？

---

> **下一步**: Day 2 将学习 LSTM 门控机制，了解如何通过精心设计的门控结构解决今天讨论的梯度消失问题。
