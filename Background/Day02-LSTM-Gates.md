# Day 2: LSTM门控机制

> 对应论文/Notebook: Hochreiter & Schmidhuber (1997) "Long Short-Term Memory" / `03_lstm_understanding.ipynb`
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 为什么需要 LSTM

在 Day 1 中我们了解到，标准 RNN 存在严重的**梯度消失问题**：当序列很长时，早期时间步的信息几乎无法影响后期的梯度更新。这意味着 RNN 很难学习长距离依赖关系。

LSTM (Long Short-Term Memory) 的核心创新是引入了一条**记忆单元通道 (cell state)**，信息可以沿着这条通道几乎不受阻碍地流动，就像一条**梯度高速公路**。

### LSTM 的两条信息通路

LSTM 维护两个状态向量：
1. **Cell State $c_t$**：长期记忆通道，信息通过加法操作更新，梯度可以无损回传
2. **Hidden State $h_t$**：短期记忆/输出，经过 $\tanh$ 压缩后的 cell state

### 三个门 (Gates) 的直觉

LSTM 通过三个**门**来控制信息流动：

- **遗忘门 (Forget Gate)**：决定从 cell state 中丢弃哪些信息。"这个信息还有用吗？"
- **输入门 (Input Gate)**：决定将哪些新信息写入 cell state。"这个新信息值得记住吗？"
- **输出门 (Output Gate)**：决定从 cell state 中输出哪些信息作为隐藏状态。"现在需要用到哪些记忆？"

每个门都是一个 sigmoid 层，输出值在 $[0, 1]$ 之间，表示"通过"的比例。

---

## 关键公式

### LSTM 完整前向传播

给定时间步 $t$ 的输入 $x_t$、上一步的隐藏状态 $h_{t-1}$ 和 cell state $c_{t-1}$：

**Step 1: 遗忘门 (Forget Gate)**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$f_t \in (0, 1)^{d_h}$ 逐元素决定保留 $c_{t-1}$ 中多少信息。$\sigma$ 为 sigmoid 函数。

**Step 2: 输入门 (Input Gate) + 候选记忆**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$i_t$ 决定写入哪些信息，$\tilde{c}_t$ 是新的候选记忆内容。

**Step 3: Cell State 更新**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

其中 $\odot$ 表示 Hadamard 乘积（逐元素乘法）。这是 LSTM 最关键的公式——cell state 通过**加法**更新，而非乘法。

**Step 4: 输出门 (Output Gate)**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

输出门控制从 cell state 中读取哪些信息作为当前隐藏状态。

### 参数维度总结

设隐藏状态维度为 $d_h$，输入维度为 $d_x$：

| 参数 | 维度 | 说明 |
|------|------|------|
| $W_f, W_i, W_c, W_o$ | $d_h \times (d_h + d_x)$ | 四组权重矩阵 |
| $b_f, b_i, b_c, b_o$ | $d_h$ | 四组偏置向量 |

**总参数量：** $4 \times d_h \times (d_h + d_x) + 4 \times d_h = 4d_h(d_h + d_x + 1)$

这是标准 RNN 参数量的大约 **4 倍**。

### 梯度流分析

LSTM 解决梯度消失的关键在于 cell state 更新公式中的**加法结构**。

对 $c_t$ 关于 $c_{t-1}$ 求导：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t + \frac{\partial (i_t \odot \tilde{c}_t)}{\partial c_{t-1}}$$

当遗忘门 $f_t \approx 1$ 时（即网络选择记住信息），梯度接近于 1，可以无损地回传多个时间步。这就是所谓的**梯度高速公路 (gradient highway)**。

对比标准 RNN：

$$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^T \cdot \text{diag}(1 - h_t^2)$$

RNN 中梯度必须经过矩阵乘法和 $\tanh$ 导数，容易指数级衰减。

---

## 直觉理解

### 传送带比喻

将 cell state 想象为一条**传送带**：
- 传送带上的物品（信息）默认会一直向前传递
- 遗忘门像一个**筛子**，把不需要的物品从传送带上移除
- 输入门像一个**装载机**，把新物品放上传送带
- 输出门像一个**检查窗口**，查看传送带上有什么并报告给外界

### 遗忘门偏置初始化的技巧

实践中，遗忘门的偏置 $b_f$ 通常初始化为**正值**（如 1.0 或 2.0）。这样训练初期遗忘门接近 1，cell state 中的信息默认被保留。

这个技巧来自 Jozefowicz et al. (2015) 的实验发现，对 LSTM 的训练稳定性有显著帮助。

### 门控的协同工作示例

以处理句子 "The cat, which ate the fish, was happy" 为例：

1. 读到 "The cat"：输入门打开，将"主语是猫"写入 cell state
2. 读到 "which ate the fish"：遗忘门部分关闭某些位置，输入门写入从句信息；但关于主语的记忆被遗忘门保护
3. 读到 "was happy"：输出门读取 cell state 中的主语信息，正确判断"猫"是 happy 的主语

### GRU：LSTM 的简化版

GRU (Gated Recurrent Unit, Cho et al. 2014) 是 LSTM 的简化变体：
- 将遗忘门和输入门合并为一个**更新门 (update gate)**
- 取消独立的 cell state，只保留隐藏状态
- 参数量约为 LSTM 的 75%
- 在很多任务上性能与 LSTM 相当

GRU 公式：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

注意 $(1 - z_t)$ 和 $z_t$ 的互补关系——遗忘多少就记住多少，这是一个优雅的约束。

---

## Peephole Connections 变体

标准 LSTM 的门只能看到 $h_{t-1}$ 和 $x_t$，但看不到 cell state $c_{t-1}$。Peephole connections (Gers & Schmidhuber, 2000) 让门直接"窥视" cell state：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + W_{pf} \odot c_{t-1} + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + W_{pi} \odot c_{t-1} + b_i)$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + W_{po} \odot c_t + b_o)$$

注意输出门使用的是**更新后**的 $c_t$。

Peephole connections 在实践中有时有帮助，有时没有。Greff et al. (2016) 的大规模实验表明，标准 LSTM 的大多数变体并不能一致性地超越原始版本。

---

## 历史背景

### LSTM 的诞生

- **1991**：Hochreiter 在其本科论文中首次分析了 RNN 中梯度消失的问题
- **1997**：Hochreiter & Schmidhuber 发表 LSTM 论文，提出门控记忆单元的概念
- **2000**：Gers & Schmidhuber 引入遗忘门（原始 LSTM 没有遗忘门！）和 peephole connections
- **2005-2012**：LSTM 在语音识别领域取得突破
- **2013**：Graves 用 LSTM 实现手写生成
- **2014**：Sutskever, Vinyals & Le 用深层 LSTM 实现机器翻译，引爆序列到序列学习
- **2014**：Cho et al. 提出 GRU 作为简化替代方案

### 为什么 LSTM 等了这么久才成功？

LSTM 在 1997 年就被发明，但直到 2013-2014 年才真正大规模成功。原因包括：
1. **计算资源**：GPU 加速使训练大规模 LSTM 成为可能
2. **数据规模**：大规模标注数据集的出现
3. **工程优化**：CuDNN 等库对 LSTM 的高效实现
4. **遗忘门的加入**：2000 年才加入的遗忘门对性能至关重要

---

## 与其他论文的关联

| 论文/主题 | 关联 |
|-----------|------|
| **RNN 基础 (Day 1)** | LSTM 直接解决 RNN 的梯度消失问题 |
| **RNN 正则化 (Day 3)** | Dropout 等正则化技术在 LSTM 上的应用需要特殊处理 |
| **ResNet (Day 6)** | 残差连接 $F(x) + x$ 与 cell state 更新 $f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ 有相同的设计哲学——通过加法结构保持梯度流 |
| **Attention 机制 (后续)** | Attention 进一步解决了 LSTM 在很长序列上的信息瓶颈问题 |
| **Neural Turing Machine (后续)** | NTM 将 LSTM 的"读写记忆"概念扩展为外部记忆 |

### LSTM 与 ResNet 的深层联系

这是一个值得深思的关联：

- **LSTM cell state**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$（门控加法）
- **ResNet 残差**: $y = F(x) + x$（直接加法）

两者都通过**加法捷径**让梯度可以直接回传，避免了深层网络/长序列中的梯度消失。可以说 ResNet 是 LSTM 思想在空间维度上的体现。

---

## 检查点问题

完成本节学习后，尝试回答以下问题：

1. **基础理解**：LSTM 有四组权重矩阵（$W_f, W_i, W_c, W_o$），如果隐藏维度为 256，输入维度为 128，总共有多少个可训练参数？

2. **梯度分析**：解释为什么 cell state 的加法更新 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ 比乘法更新 $c_t = W \cdot c_{t-1}$ 更有利于梯度回传。

3. **门控理解**：如果遗忘门 $f_t = \mathbf{0}$，输入门 $i_t = \mathbf{1}$，此时 LSTM 退化为什么？如果 $f_t = \mathbf{1}$，$i_t = \mathbf{0}$ 呢？

4. **工程实践**：为什么遗忘门的偏置建议初始化为正值（如 1.0）？如果初始化为 0 或负值会发生什么？

5. **比较分析**：LSTM 和 GRU 的主要区别是什么？在什么情况下你会选择 GRU 而不是 LSTM？

---

> **下一步**: Day 3 将学习如何对 RNN/LSTM 进行正则化，包括 variational dropout 等专门为循环网络设计的技术。
