# Day 13: 膨胀卷积与 Pre-activation ResNet

> 对应论文/Notebook: Yu & Koltun 2016 "Multi-Scale Context Aggregation by Dilated Convolutions"; He et al. 2016 "Identity Mappings in Deep Residual Networks", notebooks: 11_dilated_convolutions.ipynb, 15_identity_mappings_resnet.ipynb
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 第一部分：膨胀卷积 (Dilated Convolution)

### 1. 标准卷积的感受野问题

在标准卷积中，感受野 (receptive field) 的增长是**线性的**：

- 1 层 $3 \times 3$ 卷积：感受野 = $3 \times 3$
- 2 层 $3 \times 3$ 卷积：感受野 = $5 \times 5$
- 3 层 $3 \times 3$ 卷积：感受野 = $7 \times 7$
- $L$ 层 $3 \times 3$ 卷积：感受野 = $(2L+1) \times (2L+1)$

要获得大的感受野，传统方法有两种：

1. **增大卷积核**：参数量急剧增加（$k^2$ 增长）
2. **池化/下采样**：丢失空间分辨率

两种方法都有明显缺陷。膨胀卷积提供了第三种选择。

### 2. 膨胀卷积的定义

膨胀卷积（也称空洞卷积 / Atrous Convolution）通过在卷积核元素之间插入"空洞"来扩大感受野：

对于一维信号，标准卷积定义为：

$$(F * k)(p) = \sum_{s+t=p} F(s) \cdot k(t)$$

膨胀卷积定义为：

$$(F *_l k)(p) = \sum_{s+lt=p} F(s) \cdot k(t)$$

其中 $l$ 是**膨胀率** (dilation rate)。当 $l = 1$ 时，退化为标准卷积。

对于二维卷积，膨胀率为 $l$ 的 $k \times k$ 卷积核，其有效感受野大小为：

$$k_{\text{eff}} = k + (k - 1)(l - 1)$$

例如，$3 \times 3$ 卷积核配合不同膨胀率：

| 膨胀率 $l$ | 有效感受野 | 参数量 |
|-----------|-----------|--------|
| 1 | $3 \times 3$ | 9 |
| 2 | $5 \times 5$ | 9 |
| 4 | $9 \times 9$ | 9 |
| 8 | $17 \times 17$ | 9 |

关键洞察：**参数量不变，但感受野指数级增长**。

### 3. 膨胀率设计原则

在多层膨胀卷积中，膨胀率的选择至关重要。Yu & Koltun (2016) 提出了**指数增长策略**：

$$l_i = 2^{i-1}, \quad i = 1, 2, 3, \dots$$

即膨胀率序列为 $[1, 2, 4, 8, 16, \dots]$。

这种设计的优势：

- **感受野指数增长**：$n$ 层后感受野大小为 $O(2^n)$
- **无盲点覆盖**：每个输入位置都被至少一个卷积核元素覆盖

**网格效应 (Gridding Artifact)**：

如果膨胀率选择不当（如所有层都用相同的大膨胀率），卷积核只采样特定网格上的点，导致：
- 丢失局部细节信息
- 输出存在棋盘格状的伪影

解决方案：
- 使用递增的膨胀率序列（如 $[1, 2, 4, 8]$）
- 在膨胀卷积后接一层标准卷积（$l=1$）
- 使用混合膨胀卷积 (Hybrid Dilated Convolution)，如 $[1, 2, 5, 1, 2, 5]$

### 4. 膨胀卷积的感受野计算

对于 $n$ 层膨胀率为 $[l_1, l_2, \dots, l_n]$ 的 $3 \times 3$ 卷积：

$$\text{RF} = 1 + 2 \sum_{i=1}^{n} l_i$$

例如，$[1, 2, 4, 8]$ 四层膨胀卷积的感受野：

$$\text{RF} = 1 + 2(1 + 2 + 4 + 8) = 31$$

相比之下，4 层标准 $3 \times 3$ 卷积的感受野仅为 $9$。

### 5. 膨胀卷积的应用场景

**语义分割**：
- 需要大感受野来理解全局上下文
- 同时需要保持像素级空间分辨率
- DeepLab 系列 (Chen et al.) 大量使用膨胀卷积

**音频生成**：
- WaveNet (van den Oord et al., 2016) 使用因果膨胀卷积
- 膨胀率序列 $[1, 2, 4, \dots, 512]$，感受野覆盖数千个采样点
- 是膨胀卷积最成功的应用之一

**时序预测**：
- TCN (Temporal Convolutional Network) 使用膨胀卷积处理长时序依赖

---

### 第二部分：Pre-activation ResNet

### 6. 原始 ResNet 回顾

He et al. (2015) 提出的 ResNet 引入了残差连接：

$$y = F(x, \{W_i\}) + x$$

其中 $F$ 是残差函数。原始 ResNet 的残差块结构 (Post-activation) 为：

$$\text{Conv} \to \text{BN} \to \text{ReLU} \to \text{Conv} \to \text{BN} \to (+x) \to \text{ReLU}$$

即 **先卷积，后归一化，再激活**，最后在相加之后再做一次 ReLU。

### 7. Pre-activation 残差块

He et al. (2016) 在 "Identity Mappings in Deep Residual Networks" 中提出了改进版本：

$$\text{BN} \to \text{ReLU} \to \text{Conv} \to \text{BN} \to \text{ReLU} \to \text{Conv} \to (+x)$$

即 **先归一化，先激活，后卷积**。核心变化是将 BN 和 ReLU 移到卷积之前。

### 8. Pre-activation 的理论优势

**恒等映射的纯净性**：

在 Post-activation 中，残差块的输出为：

$$x_{l+1} = \text{ReLU}(x_l + F(x_l))$$

最后的 ReLU 使得跳跃连接不再是纯粹的恒等映射——它会截断负值。

在 Pre-activation 中：

$$x_{l+1} = x_l + F(\text{BN}(\text{ReLU}(x_l)))$$

跳跃连接传递的是**未经修改的 $x_l$**，这是一个纯粹的恒等映射。

**梯度传播分析**：

对于一个 $L$ 层的 Pre-activation ResNet，任意层 $l$ 的输出可以写为：

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$$

对损失函数 $\mathcal{L}$ 求梯度：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\right)$$

关键观察：

- 梯度中包含一个常数项 **1**（来自恒等映射）
- 这意味着梯度可以**不经过任何权重层**直接从损失函数传播到任意浅层
- 这避免了梯度消失问题，使得训练极深网络成为可能

### 9. BN 在 Pre-activation 中的新角色

在 Pre-activation 中，BN 位于卷积之前：

- **Post-activation 中的 BN**：归一化卷积的输出（在残差分支内）
- **Pre-activation 中的 BN**：归一化残差块的**输入**

这带来了一个微妙但重要的变化：

Pre-activation 的 BN 起到了**正则化**的作用——它在信息进入残差分支之前先进行归一化，使得每个残差块接收到的输入分布更稳定。

### 10. 各种残差块变体的对比

He et al. (2016) 实验了多种变体：

| 变体 | 结构 | CIFAR-10 错误率 |
|------|------|----------------|
| Post-activation (原始) | Conv→BN→ReLU→Conv→BN→(+x)→ReLU | 6.37% |
| Pre-activation (full) | BN→ReLU→Conv→BN→ReLU→Conv→(+x) | **5.71%** |
| BN after addition | Conv→Conv→(+x)→BN→ReLU | 8.17% |
| ReLU before addition | Conv→BN→ReLU→Conv→BN→ReLU→(+x) | 7.84% |

Pre-activation 在 1001 层 ResNet 上的优势尤为明显，因为更深的网络更依赖纯粹的恒等映射路径。

---

## 关键公式

### 膨胀卷积

**一维膨胀卷积**：

$$(F *_l k)(p) = \sum_{s+lt=p} F(s) \cdot k(t)$$

**有效感受野**：

$$k_{\text{eff}} = k + (k-1)(l-1) = l(k-1) + 1$$

**多层膨胀卷积感受野**（$3 \times 3$ 卷积核）：

$$\text{RF} = 1 + 2\sum_{i=1}^{n} l_i$$

### Pre-activation ResNet

**Post-activation 残差块**：

$$y_l = h(x_l) + F(x_l, W_l)$$

$$x_{l+1} = f(y_l)$$

其中 $h(x_l) = x_l$（恒等映射），$f = \text{ReLU}$（非恒等！）

**Pre-activation 残差块**：

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

其中 $\hat{f}$ 包含 BN 和 ReLU，作用在 $F$ 的输入上。

**梯度直通路径**：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} + \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F_i$$

第一项 $\frac{\partial \mathcal{L}}{\partial x_L}$ 直接从最后一层传到第 $l$ 层，不经过任何参数化变换。

---

## 直觉理解

### 膨胀卷积：用稀疏采样扩大视野

想象你站在高处俯瞰城市：
- **标准卷积**：你用望远镜看到一个很小的区域，但看得很清楚
- **大核卷积**：你换了一个更大的望远镜，看得更广但设备更重
- **池化 + 卷积**：你爬得更高，看得更广但细节模糊了
- **膨胀卷积**：你还是用同样的望远镜，但每隔几个街区采样一次——覆盖范围大了，设备没变重，而且你仍然站在原来的高度（保持分辨率）

### Pre-activation：让高速公路畅通无阻

ResNet 的跳跃连接就像高速公路——让信息（和梯度）可以快速通过。

- **Post-activation**：高速公路上有一个收费站（ReLU），虽然大部分车可以通过，但负值信号被完全阻挡
- **Pre-activation**：高速公路完全畅通（纯恒等映射），收费站被移到了辅道（残差分支）的入口

这个看似微小的改变，对于极深的网络（如 1001 层）有巨大影响——因为信号和梯度需要穿过的"收费站"越少，传播效率越高。

### 为什么 Pre-activation 在浅层网络上优势不明显？

对于较浅的网络（如 18 层或 34 层），梯度传播路径较短，Post-activation 中的 ReLU 引入的阻碍影响有限。但当网络深度增加到 100 层乃至 1000 层时，每一层的微小阻碍会累积，Pre-activation 的优势就变得显著。

---

## 历史背景

### 膨胀卷积的起源

- **1987**：Holschneider et al. 在小波变换中引入"算法 a trous"（带空洞的算法）
- **2015**：Chen et al. 将膨胀卷积引入深度学习，用于语义分割 (DeepLab)
- **2016**：Yu & Koltun 系统性地提出多尺度膨胀卷积用于密集预测
- **2016**：van den Oord et al. 在 WaveNet 中使用因果膨胀卷积
- **2017**：DeepLab v2/v3 广泛使用 ASPP (Atrous Spatial Pyramid Pooling)

### Pre-activation ResNet 的背景

- **2015 年 12 月**：He et al. 发表原始 ResNet (Post-activation)，赢得 ImageNet 2015
- **2016 年 3 月**：He et al. 发表 "Identity Mappings"，提出 Pre-activation 变体
- **2016**：Wide ResNet (Zagoruyko & Komodakis) 探索宽度 vs 深度的权衡
- **2017**：DenseNet (Huang et al.) 将残差连接推广为密集连接

Pre-activation ResNet 的提出是对残差学习理论的深化。原始 ResNet 论文主要从经验角度论证了残差连接的有效性，而 "Identity Mappings" 论文从**梯度流分析**的角度给出了理论解释，并据此提出了改进设计。

### 两者的融合

膨胀卷积和 ResNet 经常在现代架构中结合使用：

- DeepLab v3+：ResNet 骨干网络 + 膨胀卷积的空间金字塔池化
- TCN：残差连接 + 膨胀卷积处理时序数据

---

## 与其他论文的关联

### 与 Transformer (Day 10) 的关系

- Transformer 使用残差连接 + 层归一化，与 Pre-activation ResNet 的设计思想一致
- Transformer 的 Pre-Norm 变体 (LayerNorm→Attention→+x) 直接对应 Pre-activation 的思想
- 膨胀卷积试图解决的"长距离依赖"问题，在 Transformer 中通过 Self-Attention 的 $O(1)$ 路径长度来解决

### 与 GNN (Day 11) 的关系

- GNN 中的残差连接借鉴 ResNet，用于缓解过平滑
- Pre-activation 的梯度优势分析同样适用于深层 GNN
- GNN 的感受野与膨胀卷积类似，通过层数控制消息传播的范围

### 与 Bahdanau 注意力 (Day 8) 的关系

膨胀卷积和注意力机制都试图解决"如何高效获取远距离信息"的问题：
- 膨胀卷积：通过稀疏采样扩大局部操作的范围
- 注意力：通过全局加权求和直接访问所有位置

### 与 Seq2Seq for Sets (Day 12) 的关系

卷积操作（标准和膨胀）依赖于空间位置，不具有排列不变性。但 ResNet 中的全局平均池化 (GAP) 在空间维度上实现了排列不变的聚合，与 DeepSets 的求和操作类似。

---

## 检查点问题

1. **膨胀率为 4 的 $3 \times 3$ 卷积核，其有效感受野大小是多少？参数量是多少？**
   > 提示：$k_{\text{eff}} = k + (k-1)(l-1)$，参数量与膨胀率无关。

2. **为什么膨胀率通常选择指数增长（如 $1, 2, 4, 8$）？如果所有层都用相同的膨胀率会怎样？**
   > 提示：考虑网格效应和感受野的连续覆盖。

3. **Pre-activation 和 Post-activation 残差块的核心区别是什么？画出两者的结构图。**
   > 提示：BN 和 ReLU 的位置在卷积之前还是之后。

4. **为什么 Pre-activation 在极深网络（如 1001 层）上优势更明显？**
   > 提示：梯度直通路径中常数项 1 的重要性。

5. **膨胀卷积的"网格效应"是什么？如何避免？**
   > 提示：卷积核只采样特定网格上的点。

6. **如果将 Pre-activation ResNet 的跳跃连接从加法改为乘法（$x_{l+1} = x_l \cdot (1 + F(x_l))$），梯度传播会有什么变化？**
   > 提示：乘法的梯度是什么？与加法的梯度对比。

---

> **下一步**：完成理论阅读后，请分别打开 `11_dilated_convolutions.ipynb` 和 `15_identity_mappings_resnet.ipynb` 进行代码实践。
