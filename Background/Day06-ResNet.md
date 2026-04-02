# Day 6: ResNet与跳跃连接

> 对应论文/Notebook: He et al. (2015) "Deep Residual Learning for Image Recognition" / `10_resnet_deep_residual.ipynb`
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 深度网络的退化问题

在 ResNet 之前，研究者们面临一个令人困惑的现象：

简单地增加网络深度**并不总是**带来更好的性能。具体来说，一个 56 层的网络在训练集和测试集上的表现都**不如** 20 层的网络。

这**不是**过拟合问题（过拟合应该表现为训练精度高但测试精度低）。56 层网络在训练集上的精度也更低，说明这是一个**优化问题**——更深的网络更难训练。

理论上，56 层网络至少应该与 20 层网络一样好（因为多出的 36 层可以学习恒等映射）。但实践中，优化器很难学到恒等映射。

### 残差学习的核心思想

He et al. 的关键洞察：

> **学习残差映射 $F(x) = H(x) - x$ 比直接学习目标映射 $H(x)$ 更容易。**

如果期望的映射是恒等映射 $H(x) = x$，那么：
- 直接学习 $H(x) = x$ 需要网络将输出精确匹配输入（困难）
- 学习残差 $F(x) = H(x) - x = 0$ 只需将权重推向零（简单，权重初始化本身就接近零）

### 跳跃连接 (Skip Connection)

残差学习通过**跳跃连接**实现：

$$y = F(x) + x$$

输入 $x$ 直接"跳过"中间层，与 $F(x)$ 的输出相加。$F(x)$ 只需学习偏差/残差，而非完整映射。

---

## 关键公式

### 基本残差块

$$y = F(x, \{W_i\}) + x$$

对于包含两层卷积的残差块：

$$F(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

其中 $\sigma$ 是 ReLU 激活函数。最终输出：

$$y = \sigma(F(x) + x)$$

注意 ReLU 在加法之后应用。

### 维度匹配

当 $F(x)$ 和 $x$ 的维度不同时（通常在下采样时），需要投影：

$$y = F(x) + W_s \cdot x$$

其中 $W_s$ 是 $1 \times 1$ 卷积，用于调整通道数和/或空间尺寸。

论文中提出了三种处理方式：
- **(A)** 零填充增加通道数（无额外参数）
- **(B)** 仅在维度不匹配时使用投影
- **(C)** 所有跳跃连接都使用投影

实验表明 (B) 和 (C) 效果相近，(A) 略差。实践中 (B) 最常用。

### 梯度流分析

残差连接的关键优势在于梯度的传播。

对于 $L$ 个残差块的网络，第 $l$ 层到第 $L$ 层的前向传播：

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

对损失 $\mathcal{L}$ 关于 $x_l$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i, W_i)\right)$$

关键观察：梯度中始终包含一个 **$1$** 项。这意味着即使残差项 $\frac{\partial}{\partial x_l} \sum F$ 很小，梯度仍然可以通过恒等捷径直接回传。

对比普通网络：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i}$$

连乘结构导致梯度指数级衰减或爆炸（与 RNN 的问题完全相同）。

### Bottleneck 结构

对于更深的网络（ResNet-50/101/152），使用 Bottleneck 残差块以减少计算量：

```
输入 (256 通道)
    ↓
1×1 Conv, 64 通道   ← 降维（减少计算量）
    ↓ ReLU
3×3 Conv, 64 通道   ← 核心卷积（在低维空间操作）
    ↓ ReLU
1×1 Conv, 256 通道  ← 升维（恢复通道数）
    ↓
  + 输入 (跳跃连接)
    ↓ ReLU
输出
```

**参数量对比**（输入/输出 256 通道）：

| 结构 | 卷积操作 | 参数量 |
|------|---------|--------|
| 基本块 | 两个 $3 \times 3 \times 256 \times 256$ | $2 \times 3^2 \times 256^2 = 1,179,648$ |
| Bottleneck | $1 \times 1 \times 256 \times 64$ + $3 \times 3 \times 64 \times 64$ + $1 \times 1 \times 64 \times 256$ | $16,384 + 36,864 + 16,384 = 69,632$ |

Bottleneck 的参数量约为基本块的 **1/17**，计算效率显著提升。

---

## 直觉理解

### 残差学习的几何解释

在参数空间中：
- **普通网络**：优化器需要在复杂的损失曲面上找到一个"好"的点
- **残差网络**：损失曲面被"平滑化"了。每个残差块只需做微小的调整，整体路径更容易优化

可以将残差学习类比为**渐进式画画**：
- 普通网络：从白纸直接画出最终作品（困难）
- 残差网络：先有一个粗略的底稿（恒等映射），然后每一层只添加细节（残差）

### 残差连接与 LSTM 门控的共同点

这是 Sutskever 推荐论文列表中一个重要的跨领域联系：

| 特性 | LSTM | ResNet |
|------|------|--------|
| 信息传递方式 | Cell state 加法更新 | 残差加法连接 |
| 梯度保护 | $\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1$ | $\frac{\partial x_L}{\partial x_l}$ 包含常数 1 |
| 核心公式 | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | $y = F(x) + x$ |
| 设计哲学 | 信息默认流过，门控决定改变 | 信息默认流过，残差决定改变 |

两者都遵循同一原则：**让信息和梯度有一条畅通无阻的捷径**。

### 集成模型视角

Veit et al. (2016) 提出了一个有趣的视角：ResNet 可以被视为许多不同深度路径的**隐式集成**。

对于一个 3 块的 ResNet：
$$y = (F_3 + I)(F_2 + I)(F_1 + I)(x)$$
展开后：
$$y = F_3 F_2 F_1(x) + F_3 F_2(x) + F_3 F_1(x) + F_2 F_1(x) + F_3(x) + F_2(x) + F_1(x) + x$$

$n$ 个残差块产生 $2^n$ 条路径。实验表明，网络主要依赖中等长度的路径，而非最长路径。

### 为什么不用简单的拼接 (concatenation)？

与 DenseNet 的拼接方式相比，ResNet 的加法有以下特点：

| 操作 | 加法 ($y = F(x) + x$) | 拼接 ($y = [F(x), x]$) |
|------|----------------------|------------------------|
| 维度变化 | 不变 | 线性增长 |
| 内存消耗 | 较少 | 较多 |
| 信息融合 | 直接融合 | 延迟到后续层 |
| 梯度传播 | 直接且高效 | 需要学习融合 |

---

## Batch Normalization 在 ResNet 中的作用

### BN 的核心公式

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B$ 和 $\sigma_B^2$ 是 mini-batch 的均值和方差，$\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数。

### BN 在 ResNet 中的位置

原始 ResNet 使用 **post-activation** BN：

$$y = \text{ReLU}(\text{BN}(W \cdot x) + \text{shortcut})$$

后来 He et al. (2016) 在 "Identity Mappings in Deep Residual Networks" 中提出 **pre-activation** BN 更优：

$$y = W \cdot \text{ReLU}(\text{BN}(x)) + \text{shortcut}$$

Pre-activation 的优势：跳跃连接上的信号不经过任何非线性变换，梯度可以更纯粹地回传。

### BN 为何对 ResNet 至关重要

1. **稳定训练**：每层输入的分布被归一化，允许更大的学习率
2. **隐式正则化**：mini-batch 统计量引入随机性，类似 Dropout 效果
3. **加速收敛**：减少了对初始化的敏感性
4. **启用更深网络**：没有 BN，即使有残差连接，152 层网络也很难训练

---

## ResNet 架构家族

| 模型 | 层数 | 参数量 | Top-5 错误率 | 块类型 |
|------|------|--------|-------------|--------|
| ResNet-18 | 18 | 11.7M | ~10.9% | 基本块 |
| ResNet-34 | 34 | 21.8M | ~9.5% | 基本块 |
| ResNet-50 | 50 | 25.6M | ~7.7% | Bottleneck |
| ResNet-101 | 101 | 44.5M | ~7.1% | Bottleneck |
| ResNet-152 | 152 | 60.2M | ~6.7% | Bottleneck |

注意 ResNet-50 的参数量仅比 ResNet-34 多 18%，但性能提升显著——这归功于 Bottleneck 结构的高效性。

---

## 历史背景

### 深度网络训练的进化

- **2010**：Glorot & Bengio 分析了深度网络的训练困难，提出 Xavier 初始化
- **2012**：AlexNet（8 层）证明深度有用
- **2014**：VGGNet（19 层）、GoogLeNet（22 层）进一步推进深度
- **2015**：Batch Normalization 使 30+ 层网络训练成为可能
- **2015**：ResNet（152 层）通过残差连接实现质的突破，赢得 ILSVRC 2015
- **2016**：Identity Mappings in ResNet 改进了 BN 的位置（对应 notebook `15_identity_mappings_resnet.ipynb`）
- **2017**：DenseNet 将跳跃连接推向极致（每层与所有之前的层连接）

### ResNet 的影响力

ResNet 论文是计算机视觉领域被引用最多的论文之一（超过 10 万次引用）。残差连接已经成为几乎所有现代架构的标配：
- **Transformer** 中每个子层都有残差连接
- **U-Net** 中编码器和解码器之间的跳跃连接
- **DenseNet** 将残差连接推广为稠密连接
- **ResNeXt** 引入分组卷积

---

## 与其他论文的关联

| 论文/主题 | 关联 |
|-----------|------|
| **LSTM (Day 2)** | LSTM 的 cell state 和 ResNet 的跳跃连接遵循同一设计哲学 |
| **CNN/AlexNet (Day 5)** | ResNet 在 AlexNet 的基础上解决了深度训练问题 |
| **Identity Mappings (后续)** | 对 ResNet 中 BN 和激活函数位置的改进研究，`15_identity_mappings_resnet.ipynb` |
| **Dilated Convolutions (后续)** | 在残差块中使用空洞卷积扩大感受野 |
| **Attention/Transformer (后续)** | Transformer 架构中大量使用残差连接 |
| **GPipe (后续)** | 训练超大 ResNet 的流水线并行方法，`09_gpipe.ipynb` |

### 统一视角：信息高速公路

从 LSTM 到 ResNet 再到 Transformer，一条主线贯穿始终：

```
LSTM (1997)          →  ResNet (2015)        →  Transformer (2017)
c_t = f⊙c + i⊙c̃      y = F(x) + x             y = Attn(x) + x
时间维度的梯度高速公路   空间维度的梯度高速公路    序列维度的梯度高速公路
```

Ilya Sutskever 推荐这些论文的深意之一，就是理解这种跨领域的**设计模式统一性**。

---

## 检查点问题

完成本节学习后，尝试回答以下问题：

1. **退化问题**：为什么 56 层的普通网络比 20 层的性能更差？这与过拟合有什么区别？为什么简单地将多余的层设为恒等映射在实践中行不通？

2. **梯度分析**：写出经过 $n$ 个残差块后梯度的表达式。解释为什么其中的常数项 1 对于训练至关重要。

3. **Bottleneck 设计**：为什么 $1 \times 1 \to 3 \times 3 \to 1 \times 1$ 的 Bottleneck 结构比两个 $3 \times 3$ 卷积更高效？在什么情况下应该选择 Bottleneck 而非基本块？

4. **跨领域联系**：用一句话概括 LSTM 的 cell state 更新和 ResNet 的残差连接的共同设计原则。

5. **批判性思考**：ResNet 论文声称"学习残差比学习完整映射更容易"。这个说法有严格的理论证明吗？还是更多是经验观察？如果没有证明，你认为可以从哪些角度尝试解释？

---

> **Week 1 完成！** 回顾本周内容：RNN → LSTM → RNN正则化 → 网络剪枝/MDL → CNN/AlexNet → ResNet。核心主线是**如何训练更深更强的网络**，从梯度消失的诊断到门控/残差的解决方案。
