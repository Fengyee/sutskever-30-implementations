# Day 5: CNN基础与AlexNet

> 对应论文/Notebook: Krizhevsky et al. (2012) "ImageNet Classification with Deep Convolutional Neural Networks" / `07_alexnet_cnn.ipynb`
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 卷积运算的本质

卷积神经网络 (CNN) 的核心思想源于两个关键观察：

1. **局部性 (Locality)**：图像中的重要特征（边缘、纹理、物体部件）通常是局部的
2. **平移不变性 (Translation Invariance)**：同样的特征可以出现在图像的任何位置

卷积运算通过**共享权重的局部连接**完美地捕捉了这两个特性。

### 从全连接到卷积

考虑一个 $224 \times 224 \times 3$ 的彩色图像输入全连接层：
- 输入维度：$224 \times 224 \times 3 = 150,528$
- 如果全连接到 4096 个神经元：$150,528 \times 4,096 = 616,562,688$ 参数

而一个 $3 \times 3$ 的卷积核只需 $3 \times 3 \times 3 = 27$ 个参数（加偏置 28 个），就能扫描整个图像。这就是卷积网络参数高效的根本原因。

### AlexNet 的里程碑意义

AlexNet 在 2012 年 ImageNet 大规模视觉识别挑战赛 (ILSVRC) 上以压倒性优势获胜，将 Top-5 错误率从 26.2% 降低到 15.3%。这一结果震惊了计算机视觉领域，标志着**深度学习革命**的正式开始。

---

## 关键公式

### 二维离散卷积

对于单通道输入 $I$ 和卷积核 $K$，卷积运算定义为：

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$

实际上，深度学习中使用的是**互相关 (cross-correlation)**，而非严格的数学卷积（不翻转卷积核），但习惯上仍称为"卷积"。

### 多通道卷积

对于 $C_{in}$ 通道的输入和 $C_{out}$ 个卷积核：

$$Y(c_{out}, i, j) = b_{c_{out}} + \sum_{c_{in}=1}^{C_{in}} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} W(c_{out}, c_{in}, m, n) \cdot X(c_{in}, i+m, j+n)$$

每个卷积核的参数量：$C_{in} \times K_h \times K_w$，总参数量：$C_{out} \times (C_{in} \times K_h \times K_w + 1)$。

### 特征图尺寸计算

这是 CNN 中最常用的公式之一：

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

其中：
- $O$：输出特征图的边长
- $I$：输入特征图的边长
- $K$：卷积核大小
- $P$：填充 (padding) 大小
- $S$：步长 (stride)

**常见配置：**
- "Same" padding：$P = \lfloor K/2 \rfloor$，步长 $S=1$ 时输出与输入尺寸相同
- "Valid" padding：$P = 0$，输出尺寸 $= I - K + 1$

### 感受野 (Receptive Field) 计算

感受野是输出特征图上一个像素在原始输入上对应的区域大小。

对于 $L$ 层网络，第 $l$ 层的感受野大小：

$$r_l = r_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i$$

其中 $k_l$ 是第 $l$ 层的卷积核大小，$s_i$ 是第 $i$ 层的步长。

简化版（所有步长为 1 时）：

$$r_L = 1 + \sum_{l=1}^{L} (k_l - 1)$$

例如，3 层 $3 \times 3$ 卷积的感受野 $= 1 + 3 \times 2 = 7$。

这意味着两层 $3 \times 3$ 卷积（感受野 5）等效于一层 $5 \times 5$ 卷积的感受野，但参数量更少：$2 \times 3^2 = 18$ vs $5^2 = 25$。

### 池化操作

**最大池化 (Max Pooling)：**

$$Y(i, j) = \max_{0 \leq m < k, \; 0 \leq n < k} X(i \cdot s + m, \; j \cdot s + n)$$

最大池化无参数，仅做下采样和特征选择。

**平均池化 (Average Pooling)：**

$$Y(i, j) = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X(i \cdot s + m, \; j \cdot s + n)$$

---

## AlexNet 架构详解

### 完整架构

| 层 | 操作 | 输出尺寸 | 参数量 |
|----|------|---------|--------|
| 输入 | - | 227×227×3 | - |
| Conv1 | 11×11, stride 4, 96 filters | 55×55×96 | 34,944 |
| MaxPool1 | 3×3, stride 2 | 27×27×96 | 0 |
| Conv2 | 5×5, pad 2, 256 filters | 27×27×256 | 614,656 |
| MaxPool2 | 3×3, stride 2 | 13×13×256 | 0 |
| Conv3 | 3×3, pad 1, 384 filters | 13×13×384 | 885,120 |
| Conv4 | 3×3, pad 1, 384 filters | 13×13×384 | 1,327,488 |
| Conv5 | 3×3, pad 1, 256 filters | 13×13×256 | 884,992 |
| MaxPool3 | 3×3, stride 2 | 6×6×256 | 0 |
| FC6 | 4096 neurons | 4096 | 37,752,832 |
| FC7 | 4096 neurons | 4096 | 16,781,312 |
| FC8 | 1000 (classes) | 1000 | 4,097,000 |
| **总计** | | | **~62M** |

### 全连接层 vs 卷积层的参数量对比

从上表可以清楚地看到一个关键事实：

- **卷积层参数量**：约 3.7M（占总量的 ~6%）
- **全连接层参数量**：约 58.6M（占总量的 ~94%）

全连接层是参数瓶颈。后续的架构改进（如 GoogLeNet, ResNet）都在尝试减少或替代全连接层。

### AlexNet 的关键创新

**1. ReLU 激活函数**

$$\text{ReLU}(x) = \max(0, x)$$

在此之前，sigmoid 和 tanh 是主流。ReLU 的优势：
- **计算简单**：无需指数运算
- **缓解梯度消失**：正区间梯度恒为 1
- **稀疏激活**：约 50% 的神经元输出为 0，提供隐式正则化

论文中报告 ReLU 使训练速度提升约 6 倍。

**2. Local Response Normalization (LRN)**

$$b_{i}^{x,y} = \frac{a_{i}^{x,y}}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a_{j}^{x,y})^2 \right)^\beta}$$

LRN 模拟了生物神经系统中的**侧抑制 (lateral inhibition)**——活跃的神经元抑制相邻神经元。

论文中使用 $k=2, n=5, \alpha=10^{-4}, \beta=0.75$。

注意：LRN 在后续实践中被证明效果有限，已被 Batch Normalization 取代。

**3. Overlapping Pooling**

传统池化使用不重叠的窗口（如 $2 \times 2$，stride 2）。AlexNet 使用**重叠池化**（$3 \times 3$，stride 2），论文报告 Top-1 错误率降低 0.4%，Top-5 降低 0.3%。

**4. 数据增强**

- 随机裁剪 $224 \times 224$ 区域（从 $256 \times 256$ 图像中）
- 水平翻转
- PCA 颜色增强：$I_{xy} = [I_{xy}^R, I_{xy}^G, I_{xy}^B]^T + [p_1, p_2, p_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$

**5. Dropout**

在 FC6 和 FC7 层使用 $p=0.5$ 的 Dropout。这是 Dropout 首次在大规模 CNN 中成功应用。

**6. 双 GPU 训练**

由于当时 GPU 内存有限（GTX 580, 3GB），AlexNet 将网络分布在两块 GPU 上，每块 GPU 处理一半的特征图。这是早期模型并行的实践。

---

## 直觉理解

### CNN 学到了什么？

可视化 CNN 的不同层揭示了层次化的特征学习：

| 层级 | 学到的特征 | 类比 |
|------|-----------|------|
| 浅层 (Conv1-2) | 边缘、颜色、纹理 | 笔画 |
| 中层 (Conv3-4) | 纹理组合、局部形状 | 部件 |
| 深层 (Conv5) | 物体部件、语义特征 | 物体组件 |
| 全连接层 | 全局语义、类别相关特征 | 整体识别 |

这种层次化表示与人类视觉皮层的结构惊人地相似（V1 → V2 → V4 → IT cortex）。

### 为什么卷积比全连接好？

对于图像任务，卷积网络有三个关键归纳偏置 (inductive bias)：

1. **局部连接**：每个神经元只看输入的局部区域，减少参数量
2. **参数共享**：同一个卷积核在所有空间位置复用，进一步减少参数
3. **平移等变性**：$f(shift(x)) = shift(f(x))$，对平移操作具有自然的不变性

这些偏置与图像的物理特性完美匹配，使得 CNN 在视觉任务上远超全连接网络。

### 1×1 卷积的威力

虽然 AlexNet 没有使用 $1 \times 1$ 卷积，但后续架构（GoogLeNet, ResNet）大量使用。$1 \times 1$ 卷积的作用：

- 跨通道信息融合（不改变空间尺寸）
- 通道数升维/降维（控制计算量）
- 引入非线性（后接 ReLU）

参数量：$C_{in} \times C_{out}$，类似于在每个空间位置做一个全连接变换。

---

## 历史背景

### CNN 的前世今生

- **1980**：福岛邦彦提出 Neocognitron，CNN 的雏形
- **1989**：LeCun et al. 提出 LeNet，首次将反向传播应用于卷积网络
- **1998**：LeNet-5 在手写数字识别上取得成功
- **1999-2011**：CNN 研究陷入低谷，SVM 和手工特征主导计算机视觉
- **2012**：AlexNet 横空出世，深度学习革命开始
- **2013**：ZFNet（Zeiler & Fergus）微调 AlexNet，提供特征可视化
- **2014**：VGGNet 证明深度的重要性（16-19 层），GoogLeNet 引入 Inception 模块
- **2015**：ResNet 突破 100 层深度，超越人类水平

### AlexNet 成功的关键因素

AlexNet 的成功不仅仅是架构创新，而是多个因素的汇聚：

1. **大规模数据**：ImageNet 提供了 120 万训练图像
2. **GPU 计算**：GPU 的并行计算能力使大规模 CNN 训练成为可能
3. **正确的架构选择**：ReLU、Dropout、数据增强的组合
4. **工程创新**：双 GPU 并行训练

---

## 与其他论文的关联

| 论文/主题 | 关联 |
|-----------|------|
| **网络剪枝 (Day 4)** | Han et al. 2015 在 AlexNet 上实现 90%+ 的剪枝率，全连接层可剪枝 96% |
| **ResNet (Day 6)** | 在 AlexNet 基础上解决了深层网络训练的退化问题 |
| **Dilated Convolutions (后续)** | 通过空洞卷积扩大感受野，不增加参数 |
| **Batch Normalization (后续)** | 取代了 AlexNet 的 LRN |
| **Transformer (后续)** | Vision Transformer 挑战了 CNN 的局部性假设 |

### 从 AlexNet 到现代架构的演进

```
AlexNet (2012) → VGGNet (2014) → GoogLeNet (2014) → ResNet (2015)
    8层              19层           22层              152层
    62M参数          138M参数       5M参数            60M参数
```

关键趋势：更深、更高效、参数利用率更高。

---

## 检查点问题

完成本节学习后，尝试回答以下问题：

1. **尺寸计算**：输入 $32 \times 32$ 的图像，经过一个 $5 \times 5$、stride 1、padding 0 的卷积层，输出尺寸是多少？如果再经过 $2 \times 2$、stride 2 的最大池化呢？

2. **参数对比**：一个 $3 \times 3 \times 64 \times 128$ 的卷积层（64 通道输入，128 个卷积核）有多少参数？等效的全连接层（假设输入特征图为 $56 \times 56 \times 64$）需要多少参数？

3. **感受野**：三层 $3 \times 3$、stride 1 卷积的感受野等效于一层多大的卷积核？哪种方案参数更少？哪种方案非线性更强？

4. **架构分析**：AlexNet 中 94% 的参数在全连接层，但大部分计算量在卷积层。解释为什么。

5. **批判性思考**：AlexNet 使用了 $11 \times 11$ 的第一层卷积核，但后续架构几乎都使用 $3 \times 3$ 或 $7 \times 7$。$11 \times 11$ 卷积核有什么问题？

---

> **下一步**: Day 6 将学习 ResNet 的残差连接，了解如何通过简单而优雅的跳跃连接训练超过 100 层的深度网络。
