# Day 15: 变分自编码器 (Variational Autoencoder, VAE)

> 对应论文/Notebook: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014); notebook: 17_variational_autoencoder.ipynb
> 学习时间: 15 分钟理论阅读

---

## 核心概念

### 1. 从自编码器到变分自编码器

传统自编码器 (AE) 将输入 $x$ 编码为潜在表示 $z$，然后从 $z$ 重建 $x$。
这个过程是**确定性的**——每个输入对应一个固定的潜在向量。

VAE 的核心突破在于：**将编码过程变为概率性的**。编码器不再输出一个点，
而是输出一个**概率分布**的参数（均值 $\mu$ 和方差 $\sigma^2$），
然后从这个分布中采样得到 $z$。

这个看似微小的改变带来了根本性的不同：

- AE 的潜在空间可能是不连续、不规则的，无法用于生成
- VAE 的潜在空间被约束为平滑、连续的，可以从中采样生成新数据

### 2. 生成模型的视角

VAE 本质上是一个**深度潜变量模型** (deep latent variable model)。
我们假设数据的生成过程如下：

1. 从先验分布 $p(z)$ 中采样潜变量 $z$（通常选择标准正态分布）
2. 从条件分布 $p_\theta(x|z)$ 中生成观测数据 $x$

我们的目标是最大化数据的边际似然：

$$p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$$

但这个积分通常是**不可解的** (intractable)，因为需要对所有可能的 $z$ 积分。

### 3. 变分推断

为了绕过不可解的积分，VAE 引入了一个**近似后验** $q_\phi(z|x)$，
用参数化的神经网络（编码器）来近似真实后验 $p_\theta(z|x)$。

这就是"变分"二字的由来——我们在一族参数化分布中寻找最佳近似。

---

## 关键公式

### ELBO（证据下界）

VAE 的核心优化目标是 **ELBO (Evidence Lower Bound)**：

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

右边即为 ELBO，由两部分组成：

| 项 | 含义 | 直觉 |
|---|---|---|
| $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ | **重建损失** | 编码后能否还原原始数据 |
| $-D_{KL}(q_\phi(z|x) \| p(z))$ | **KL 散度正则项** | 近似后验不要偏离先验太远 |

### ELBO 推导

从 $\log p_\theta(x)$ 出发：

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x)]$$

$$= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{p_\theta(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)} \cdot \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]}_{\text{ELBO}} + \underbrace{D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}_{\geq 0}$$

由于 KL 散度恒非负，所以 ELBO 确实是 $\log p_\theta(x)$ 的下界。

### KL 散度的解析形式

当 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ 且 $p(z) = \mathcal{N}(0, I)$ 时，
KL 散度有解析解：

$$D_{KL}(q_\phi(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

其中 $J$ 是潜在空间的维度。

### 重参数化技巧 (Reparameterization Trick)

采样操作 $z \sim q_\phi(z|x)$ 是不可微的，无法直接反向传播。
重参数化技巧将随机性从计算图中分离出来：

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\odot$ 表示逐元素乘法。这样，$z$ 对 $\phi$ 的梯度可以通过 $\mu$ 和 $\sigma$ 传播，
而随机性完全由外部噪声 $\epsilon$ 提供。

---

## 直觉理解

### 为什么需要重参数化

想象你在优化一个函数 $f(z)$，其中 $z$ 是从某个分布采样的。
如果采样操作本身依赖于你要优化的参数，梯度就断了——
你无法对"从分布中采样"这个操作求导。

重参数化的天才之处在于：**把采样变成了确定性变换加上固定噪声**。
这就像是说："我不是从一个会变的分布里抽样，
而是先抽一个固定的骰子结果，然后用可微函数变换它。"

### KL 正则化的直觉

KL 项的作用是鼓励潜在空间的**平滑连续性**：

- 没有 KL 项时，编码器会学到极度集中的后验（类似 delta 函数），
  不同类别的数据点在潜在空间中被分得很远，中间是"死区"
- KL 项迫使每个数据点的编码分布接近标准正态分布，
  这样不同数据点的编码区域会有重叠，潜在空间变得连续
- 结果：在潜在空间中平滑插值可以生成有意义的中间样本

### VAE vs AE 的本质区别

| 维度 | AE | VAE |
|---|---|---|
| 编码方式 | 确定性映射 | 概率分布 |
| 潜在空间 | 可能不连续 | 平滑连续 |
| 是否可生成 | 不能（无法采样） | 可以（从先验采样） |
| 优化目标 | 重建误差 | ELBO |
| 理论基础 | 无概率框架 | 变分推断 |

### 后验坍缩 (Posterior Collapse)

后验坍缩是 VAE 训练中的常见问题：

- **现象**：$q_\phi(z|x)$ 坍缩到先验 $p(z)$，KL 项变为零
- **后果**：潜变量 $z$ 不携带任何关于 $x$ 的信息，解码器忽略 $z$
- **原因**：当解码器足够强大时（如自回归解码器），
  它可以仅依靠自身的上下文建模能力生成好的输出，无需依赖 $z$
- **缓解方法**：
  - KL 退火 (KL annealing)：训练初期降低 KL 权重，逐步增加
  - Free bits：为每个维度设置最小 KL 值
  - 使用更弱的解码器

---

## 历史背景

### 2014: VAE 的诞生

Diederik P. Kingma 和 Max Welling 于 2014 年在 ICLR 上发表了
"Auto-Encoding Variational Bayes"，开创了深度生成模型的新范式。

几乎同时，Danilo Rezende、Shakir Mohamed 和 Daan Wierstra
独立提出了类似的思想（"Stochastic Backpropagation and Approximate
Inference in Deep Generative Models"）。

### 变分推断的传统

VAE 建立在**变分推断** (variational inference) 的长期传统之上。
变分推断可追溯至统计物理和贝叶斯统计，核心思想是用优化问题替代积分问题。
Michael Jordan、Tommi Jaakkola 等人在 1990 年代将其系统化。

VAE 的贡献是将变分推断与深度学习结合，
用神经网络参数化变分族，实现了高效的摊销推断 (amortized inference)。

### 后续发展

- **VQ-VAE** (2017): 使用离散潜变量，结合向量量化
- **beta-VAE** (2017): 调节 KL 权重以学习解纠缠表示
- **NVAE** (2020): 层级 VAE，达到了与 GAN 竞争的图像质量
- **Diffusion Models** (2020-): 可以看作层级 VAE 的极限形式

---

## 与其他论文的关联

| 论文 | 关联 |
|---|---|
| **GAN (Goodfellow 2014)** | 同为深度生成模型，但训练方式完全不同；GAN 用对抗训练，VAE 用变分推断 |
| **Attention (Bahdanau 2015)** | 注意力机制可以增强 VAE 的编码器和解码器 |
| **Transformer (Vaswani 2017)** | Transformer 架构可以作为 VAE 的 backbone |
| **Neural Turing Machine (Day 18)** | NTM 的记忆读写也涉及软寻址，与 VAE 的连续潜变量有理念上的相似 |
| **Information Theory (Day 20)** | VAE 的 ELBO 可以从信息论角度理解：rate-distortion trade-off |

---

## 检查点问题

1. **ELBO 的两个组成部分分别是什么？它们之间的权衡关系是什么？**
   提示：考虑重建质量 vs 潜在空间的规则性。

2. **为什么不能直接对采样操作 $z \sim q_\phi(z|x)$ 求梯度？重参数化技巧如何解决这个问题？**
   提示：考虑计算图中的随机节点。

3. **如果去掉 KL 散度项，VAE 会退化成什么？潜在空间会有什么问题？**
   提示：考虑编码分布的形状和潜在空间的连续性。

4. **什么是后验坍缩？为什么强大的解码器容易导致这个问题？**
   提示：当解码器可以独立建模 $p(x)$ 时，$z$ 的信息是否还有用。

5. **从信息论的角度，KL 项 $D_{KL}(q(z|x) \| p(z))$ 可以被理解为什么？**
   提示：想想编码 $z$ 需要多少额外的 bits。

6. **VAE 和扩散模型 (Diffusion Models) 有什么理论上的联系？**
   提示：考虑层级潜变量模型。
