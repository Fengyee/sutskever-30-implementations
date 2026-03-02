# 🎯 Sutskever 30 — 四周系统学习计划

> **目标**：4 周内系统掌握本仓库 30 篇论文的核心知识，达到可独立应用的水平。
>
> **适用对象**：计算机科学背景、具备自驱学习能力、偏好结构化和深度理解。
>
> **约束**：每天 ≤ 1 小时 | 理论 + 实践并重 | 每周自测 | 含复习间隔机制与难度递增曲线

---

## 📐 整体设计理念

### 难度递增曲线

```
难度
 ▲
 │            ┌────────────────── 第4周：综合应用与前沿
 │         ┌──┘                   (难度 8-9/10)
 │      ┌──┘   第3周：高级架构与理论
 │   ┌──┘      (难度 6-8/10)
 │┌──┘  第2周：核心架构
 ││     (难度 4-6/10)
 ├┘ 第1周：基础
 │  (难度 2-4/10)
 └──────────────────────────────────► 时间（天）
   1    7   8   14  15  21  22  28
```

### 复习间隔机制（基于间隔重复原理）

| 学习日 | 首次复习 | 二次复习 | 三次复习 |
|--------|---------|---------|---------|
| Day 1  | Day 3   | Day 7   | Day 14  |
| Day 4  | Day 7   | Day 11  | Day 18  |
| Day 8  | Day 10  | Day 15  | Day 22  |
| Day 11 | Day 14  | Day 19  | Day 26  |
| Day 15 | Day 18  | Day 23  | Day 28  |
| Day 19 | Day 22  | Day 27  | —       |

> 💡 **实践方法**：复习时不重跑整个 Notebook，而是：
> 1. 闭卷回忆该论文的核心公式与架构图（2 分钟）
> 2. 打开 Notebook 验证记忆偏差（3 分钟）
> 3. 修改一个关键参数，观察输出变化（5 分钟）

### 时间分配原则

每日 60 分钟标准分配：

| 模块 | 时长 | 内容 |
|------|------|------|
| 🧠 理论阅读 | 15 min | 论文核心思想、数学推导 |
| 💻 代码实践 | 30 min | 运行 Notebook、修改参数、观察结果 |
| 📝 复习 & 笔记 | 10 min | 间隔复习 + 记录关键洞察 |
| 🔗 关联思考 | 5 min | 连接不同论文间的概念 |

---

## 📅 第一周：基础——序列模型与训练技术

> **核心能力提升**：掌握 RNN/LSTM 的前向/反向传播机制；理解正则化与压缩的本质关系
>
> **难度区间**：⭐⭐ ~ ⭐⭐⭐⭐ (2-4/10)

### Day 1（周一）：RNN 基础 — 字符级语言模型

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | RNN 核心公式：$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$；理解 BPTT | — |
| 实践 (30min) | 运行 `02_char_rnn_karpathy.ipynb`，观察文本生成过程 | `02_char_rnn_karpathy.ipynb` |
| 笔记 (10min) | 画出 RNN 展开图，标注梯度流方向 | — |
| 关联 (5min) | 思考：为什么长序列会导致梯度消失？ | — |

**✅ 检查点**：能解释 RNN 单步前向传播的矩阵运算

### Day 2（周二）：LSTM — 门控机制

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | LSTM 四个门的公式：遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$、候选 $\tilde{c}_t$ | — |
| 实践 (30min) | 运行 `03_lstm_understanding.ipynb`，重点看门激活可视化 | `03_lstm_understanding.ipynb` |
| 笔记 (10min) | 绘制 LSTM Cell 结构图，标注各门信号流 | — |
| 关联 (5min) | 对比 RNN vs LSTM 在长序列上的表现差异 | — |

**✅ 检查点**：能手写 LSTM 单步前向传播的伪代码

### Day 3（周三）：RNN 正则化 + Day 1 复习

| 时段 | 内容 | 文件 |
|------|------|------|
| 复习 (10min) | 闭卷回忆 Day 1 RNN 公式，打开 Notebook 验证 | `02_char_rnn_karpathy.ipynb` |
| 理论 (10min) | Variational Dropout：为什么 RNN 的 Dropout 需要特殊处理？ | — |
| 实践 (30min) | 运行 `04_rnn_regularization.ipynb`，对比有/无 Dropout 的训练曲线 | `04_rnn_regularization.ipynb` |
| 笔记 (10min) | 记录 Dropout 位置对 RNN 的影响规律 | — |

**✅ 检查点**：能解释为什么普通 Dropout 不适用于 RNN

### Day 4（周四）：网络剪枝与 MDL 原则

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | MDL 原则：最短描述长度 = 最优模型；剪枝 = 压缩 | — |
| 实践 (30min) | 运行 `05_neural_network_pruning.ipynb`，观察 90%+ 剪枝后精度变化 | `05_neural_network_pruning.ipynb` |
| 笔记 (10min) | 记录剪枝率 vs 精度的关系曲线 | — |
| 关联 (5min) | 思考：压缩与泛化的关系是什么？ | — |

**✅ 检查点**：能解释迭代剪枝的基本流程

### Day 5（周五）：CNN 基础 — AlexNet

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 卷积运算、感受野、特征图；AlexNet 架构里程碑 | — |
| 实践 (30min) | 运行 `07_alexnet_cnn.ipynb`，修改卷积核大小观察效果 | `07_alexnet_cnn.ipynb` |
| 笔记 (10min) | 画出卷积运算示意图（input → filter → feature map） | — |
| 关联 (5min) | 对比全连接层 vs 卷积层的参数量差异 | — |

**✅ 检查点**：能计算给定输入/卷积核大小的输出特征图尺寸

### Day 6（周六）：ResNet — 跳跃连接

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 残差学习：$F(x) + x$；为什么恒等映射让深度网络可训练 | — |
| 实践 (30min) | 运行 `10_resnet_deep_residual.ipynb`，可视化梯度流 | `10_resnet_deep_residual.ipynb` |
| 笔记 (10min) | 对比 Plain Network vs ResNet 的训练曲线 | — |
| 关联 (5min) | 思考：残差连接和 LSTM 的门控有什么共同点？ | — |

**✅ 检查点**：能画出 Residual Block 结构并解释为什么它能解决退化问题

### Day 7（周日）：周自测 + 综合复习

| 时段 | 内容 |
|------|------|
| 复习 (20min) | 快速回顾 Day 1-6 的笔记，复习 Day 1, 4 的内容（间隔复习） |
| 自测 (40min) | 完成下方自测题 |

#### 📋 第一周自测题

1. **概念题**：画出 LSTM Cell 结构图，标注遗忘门、输入门、输出门，写出各门的数学公式
2. **对比题**：列表对比 Vanilla RNN vs LSTM 在以下维度的差异：梯度流、记忆长度、参数量、计算复杂度
3. **分析题**：为什么 90% 的权重被剪枝后模型仍能保持较好精度？用信息论/MDL 的视角解释
4. **计算题**：输入 $32 \times 32 \times 3$，卷积核 $5 \times 5$，stride=1，padding=2，输出特征图尺寸是多少？
5. **编程题**：不看代码，手写一个 RNN 单步前向传播的 NumPy 实现（10 行以内）

**🎯 及格标准**：5 题答对 3 题以上

### 📊 第一周可衡量成果

- [ ] 能独立画出 RNN 和 LSTM 的架构图
- [ ] 能解释 Variational Dropout 与标准 Dropout 的区别
- [ ] 能计算 CNN 各层的输出尺寸
- [ ] 能解释 ResNet 解决的核心问题
- [ ] 完成自测题得分 ≥ 60%

---

## 📅 第二周：核心架构——注意力机制与现代网络

> **核心能力提升**：深入理解注意力机制（从 Bahdanau → Transformer）；掌握图结构与集合处理
>
> **难度区间**：⭐⭐⭐⭐ ~ ⭐⭐⭐⭐⭐⭐ (4-6/10)

### Day 8（周一）：注意力机制 — Bahdanau Attention

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | Seq2Seq 的信息瓶颈；注意力 = 加权求和上下文向量 | — |
| 实践 (30min) | 运行 `14_bahdanau_attention.ipynb`，可视化对齐矩阵 | `14_bahdanau_attention.ipynb` |
| 笔记 (10min) | 画出 Encoder-Decoder + Attention 的完整架构 | — |
| 关联 (5min) | 对比：固定上下文向量 vs 动态注意力上下文的区别 | — |

**✅ 检查点**：能写出注意力权重的计算公式 $\alpha_{ij} = \text{softmax}(e_{ij})$

### Day 9（周二）：Pointer Networks — 注意力作为选择

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 指针网络：输出词汇 = 输入序列位置；组合优化问题 | — |
| 实践 (30min) | 运行 `06_pointer_networks.ipynb`，理解注意力如何"指向"输入 | `06_pointer_networks.ipynb` |
| 笔记 (10min) | 对比 Standard Attention vs Pointer Attention 的差异 | — |
| 关联 (5min) | 思考：指针网络如何处理可变输出大小的问题？ | — |

**✅ 检查点**：能解释指针网络与标准 Seq2Seq 在输出机制上的本质区别

### Day 10（周三）：Transformer — Self-Attention + Day 8 复习

| 时段 | 内容 | 文件 |
|------|------|------|
| 复习 (10min) | 回忆 Day 8 注意力公式，修改 Bahdanau Notebook 的一个参数 | `14_bahdanau_attention.ipynb` |
| 理论 (10min) | 自注意力 $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$；多头注意力；位置编码 | — |
| 实践 (30min) | 运行 `13_attention_is_all_you_need.ipynb`，重点看多头注意力实现 | `13_attention_is_all_you_need.ipynb` |
| 笔记 (10min) | 画出 Transformer 编码器块（Self-Attention → FFN → LayerNorm） | — |

**✅ 检查点**：能解释 $\sqrt{d_k}$ 缩放因子的作用

### Day 11（周四）：图神经网络 — 消息传递

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 消息传递框架：Aggregate → Update；GCN、GraphSAGE 核心思想 | — |
| 实践 (30min) | 运行 `12_graph_neural_networks.ipynb`，理解邻接矩阵在消息传递中的作用 | `12_graph_neural_networks.ipynb` |
| 笔记 (10min) | 画出一轮消息传递的示意图 | — |
| 关联 (5min) | 思考：GNN 的消息传递与 Transformer 的自注意力有什么联系？ | — |

**✅ 检查点**：能解释 GNN 如何处理不规则图结构数据

### Day 12（周五）：Seq2Seq for Sets — 集合的排列不变性

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 排列不变性：为什么集合编码需要特殊处理；Read-Process-Write 架构 | — |
| 实践 (30min) | 运行 `08_seq2seq_for_sets.ipynb`，对比有序 vs 无序编码的差异 | `08_seq2seq_for_sets.ipynb` |
| 笔记 (10min) | 记录 Attention-based Set Encoding 的实现要点 | — |
| 关联 (5min) | 思考：Transformer 的 Self-Attention 是否天然具有排列不变性？ | — |

**✅ 检查点**：能解释为什么传统 Seq2Seq 不适合处理集合输入

### Day 13（周六）：膨胀卷积与 Pre-activation ResNet

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (10min) | 膨胀卷积：不增加参数扩大感受野；Pre-activation：BN→ReLU→Conv | — |
| 实践 (20min) | 运行 `11_dilated_convolutions.ipynb`，观察不同膨胀率的感受野 | `11_dilated_convolutions.ipynb` |
| 实践 (20min) | 运行 `15_identity_mappings_resnet.ipynb`，对比 Pre vs Post activation | `15_identity_mappings_resnet.ipynb` |
| 笔记 (10min) | 记录膨胀率设计原则 & Pre-activation 的梯度优势 | — |

**✅ 检查点**：能解释膨胀卷积如何在不损失分辨率的情况下增大感受野

### Day 14（周日）：周自测 + 综合复习

| 时段 | 内容 |
|------|------|
| 复习 (20min) | 复习 Day 1（三次复习）、Day 4（二次复习）、Day 8（首次复习）的内容 |
| 自测 (40min) | 完成下方自测题 |

#### 📋 第二周自测题

1. **推导题**：从 Bahdanau Attention 到 Scaled Dot-Product Attention，写出关键演变步骤和公式
2. **架构题**：画出完整的 Transformer Encoder Block，标注每个子层的输入/输出维度
3. **对比题**：列表对比 CNN / RNN / GNN / Transformer 在以下维度的差异：归纳偏置、并行性、适用数据结构
4. **分析题**：Pointer Network 为什么能处理 TSP（旅行商问题）这类组合优化？它的输出空间与标准 Seq2Seq 有何不同？
5. **编程题**：不看代码，手写 Scaled Dot-Product Attention 的 NumPy 实现（15 行以内）

**🎯 及格标准**：5 题答对 3 题以上

### 📊 第二周可衡量成果

- [ ] 能独立画出 Transformer 完整架构
- [ ] 能手写 Attention 计算的 NumPy 代码
- [ ] 能解释 4 种网络架构（CNN/RNN/GNN/Transformer）各自的归纳偏置
- [ ] 能区分 Bahdanau Attention 和 Self-Attention 的异同
- [ ] 完成自测题得分 ≥ 60%

---

## 📅 第三周：进阶——高级架构与理论基础

> **核心能力提升**：掌握生成模型（VAE）、外部记忆（NTM）、关系推理；建立信息论直觉
>
> **难度区间**：⭐⭐⭐⭐⭐⭐ ~ ⭐⭐⭐⭐⭐⭐⭐⭐ (6-8/10)

### Day 15（周一）：VAE — 变分自编码器

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | ELBO = 重建损失 + KL 散度；重参数化技巧 $z = \mu + \sigma \cdot \epsilon$ | — |
| 实践 (30min) | 运行 `17_variational_autoencoder.ipynb`，可视化潜在空间 | `17_variational_autoencoder.ipynb` |
| 笔记 (10min) | 画出 VAE 的编码器-解码器架构，标注 KL 损失和重建损失 | — |
| 关联 (5min) | 思考：VAE 与 AE 的本质区别是什么？为什么需要 KL 正则化？ | — |

**✅ 检查点**：能解释重参数化技巧为什么能让梯度通过随机采样

### Day 16（周二）：关系推理 — Relation Networks

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 关系网络：$\text{RN}(O) = f_\phi(\sum_{i,j} g_\theta(o_i, o_j))$；成对推理 | — |
| 实践 (30min) | 运行 `16_relational_reasoning.ipynb`，理解 pairwise 交互如何工作 | `16_relational_reasoning.ipynb` |
| 笔记 (10min) | 对比 Relation Network vs GNN 的消息传递机制 | — |
| 关联 (5min) | 思考：Transformer 的 Self-Attention 也是一种关系推理吗？ | — |

**✅ 检查点**：能解释为什么 Relation Network 的复杂度是 $O(n^2)$

### Day 17（周三）：Relational RNN + Day 15 复习

| 时段 | 内容 | 文件 |
|------|------|------|
| 复习 (10min) | 回忆 Day 15 VAE 的 ELBO 公式，修改潜在维度观察效果 | `17_variational_autoencoder.ipynb` |
| 理论 (10min) | Relational RNN：LSTM + 关系记忆模块（多头自注意力在记忆槽上） | — |
| 实践 (30min) | 运行 `18_relational_rnn.ipynb`，理解记忆槽的交互机制 | `18_relational_rnn.ipynb` |
| 笔记 (10min) | 画出 Relational Memory Core 的结构 | — |

**✅ 检查点**：能解释 Relational RNN 如何将注意力机制引入 RNN 的记忆

### Day 18（周四）：Neural Turing Machine — 外部记忆

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | NTM：内容寻址 + 位置寻址；可微分读写操作 | — |
| 实践 (30min) | 运行 `20_neural_turing_machine.ipynb`，观察 attention 权重在记忆上的移动 | `20_neural_turing_machine.ipynb` |
| 笔记 (10min) | 对比 NTM vs LSTM vs Relational RNN 的记忆机制 | — |
| 关联 (5min) | 思考：Transformer 的 KV Cache 是否可以看作一种外部记忆？ | — |

**✅ 检查点**：能解释内容寻址和位置寻址的区别

### Day 19（周五）：CTC 损失与语音识别

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | CTC：无需对齐的序列到序列训练；前向算法和空白标签 | — |
| 实践 (30min) | 运行 `21_ctc_speech.ipynb`，理解路径折叠和前向算法 | `21_ctc_speech.ipynb` |
| 笔记 (10min) | 画出 CTC 的格状图（lattice），标注有效路径 | — |
| 关联 (5min) | 思考：CTC vs Attention-based Seq2Seq 各自的优劣是什么？ | — |

**✅ 检查点**：能解释 CTC 如何在不知道输入-输出对齐的情况下训练

### Day 20（周六）：信息论基础 — MDL 与 Kolmogorov 复杂度

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (10min) | MDL：数据压缩 = 模型选择；Kolmogorov 复杂度：最短程序长度 | — |
| 实践 (20min) | 运行 `23_mdl_principle.ipynb`，理解 MDL vs AIC/BIC 的对比 | `23_mdl_principle.ipynb` |
| 实践 (20min) | 运行 `25_kolmogorov_complexity.ipynb`，理解不可压缩性 = 随机性 | `25_kolmogorov_complexity.ipynb` |
| 笔记 (10min) | 记录 Shannon 熵 vs Kolmogorov 复杂度的关系 | — |

**✅ 检查点**：能用自己的话解释"压缩即理解"的含义

### Day 21（周日）：周自测 + 综合复习

| 时段 | 内容 |
|------|------|
| 复习 (20min) | 复习 Day 4（三次复习）、Day 8（二次复习）、Day 11（二次复习）、Day 15（首次复习）的内容 |
| 自测 (40min) | 完成下方自测题 |

#### 📋 第三周自测题

1. **推导题**：写出 VAE 的 ELBO 公式，解释每一项的含义，并推导重参数化技巧
2. **架构题**：画出 Neural Turing Machine 的完整架构，标注控制器、读写头和内存矩阵的交互
3. **对比题**：列表对比 LSTM / NTM / Relational RNN / Transformer 在"记忆"维度上的差异（记忆容量、访问方式、可扩展性）
4. **分析题**：如果一个模型在测试集上的 MDL 值比另一个模型低，为什么我们倾向于选择它？这与奥卡姆剃刀有什么关系？
5. **综合题**：选择你认为最重要的 3 篇论文，写出它们之间的技术传承关系（每篇 2-3 句）

**🎯 及格标准**：5 题答对 3 题以上

### 📊 第三周可衡量成果

- [ ] 能解释 VAE 的完整训练流程（编码 → 采样 → 解码 → 损失）
- [ ] 能对比 3 种以上的"记忆"机制（LSTM、NTM、Transformer KV Cache）
- [ ] 能解释 CTC 损失的前向算法
- [ ] 能用信息论视角分析模型选择问题
- [ ] 完成自测题得分 ≥ 60%

---

## 📅 第四周：综合——前沿应用与系统整合

> **核心能力提升**：掌握现代 LLM 核心技术栈（RAG、缩放律）；建立从理论到应用的完整认知框架
>
> **难度区间**：⭐⭐⭐⭐⭐⭐⭐⭐ ~ ⭐⭐⭐⭐⭐⭐⭐⭐⭐ (8-9/10)

### Day 22（周一）：Scaling Laws — 缩放定律

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 幂律关系：$L(N) \propto N^{-\alpha}$；计算最优训练（Chinchilla） | — |
| 实践 (30min) | 运行 `22_scaling_laws.ipynb`，拟合幂律曲线，理解预测能力 | `22_scaling_laws.ipynb` |
| 笔记 (10min) | 记录关键缩放指数和 compute-optimal 规则 | — |
| 关联 (5min) | 思考：缩放律对模型训练预算分配有什么启示？ | — |

**✅ 检查点**：能解释 Chinchilla 的核心结论（模型大小和数据量应如何平衡）

### Day 23（周二）：GPipe — 流水线并行 + Day 19 复习

| 时段 | 内容 | 文件 |
|------|------|------|
| 复习 (10min) | 回忆 Day 19 的关键概念（间隔复习） | — |
| 理论 (10min) | 流水线并行：模型分区 + 微批次；气泡时间分析 | — |
| 实践 (30min) | 运行 `09_gpipe.ipynb`，观察微批次的流水线调度 | `09_gpipe.ipynb` |
| 笔记 (10min) | 画出 F-then-B 调度的时间线图 | — |

**✅ 检查点**：能解释流水线并行如何让超大模型训练成为可能

### Day 24（周三）：Dense Passage Retrieval — 稠密检索

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | 双编码器架构；In-batch Negatives；最大内积搜索（MIPS） | — |
| 实践 (30min) | 运行 `28_dense_passage_retrieval.ipynb`，理解向量空间中的语义匹配 | `28_dense_passage_retrieval.ipynb` |
| 笔记 (10min) | 画出 DPR 的训练和推理流程图 | — |
| 关联 (5min) | 思考：稠密检索 vs 稀疏检索（BM25）的优劣 | — |

**✅ 检查点**：能解释 In-batch Negatives 为什么能提升训练效率

### Day 25（周四）：RAG — 检索增强生成

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (15min) | RAG-Sequence vs RAG-Token；检索 + 生成的端到端训练 | — |
| 实践 (30min) | 运行 `29_rag.ipynb`，理解检索如何增强生成质量 | `29_rag.ipynb` |
| 笔记 (10min) | 画出 RAG 的完整推理流程：Query → Retrieve → Generate | — |
| 关联 (5min) | 将 DPR（Day 24）和 RAG 连接起来形成完整管道 | — |

**✅ 检查点**：能解释 RAG-Sequence 和 RAG-Token 的区别

### Day 26（周五）：Lost in the Middle + Multi-token Prediction

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (10min) | 长上下文的位置偏差：U 形曲线；多 Token 预测的效率优势 | — |
| 实践 (20min) | 运行 `30_lost_in_middle.ipynb`，观察位置对检索性能的影响 | `30_lost_in_middle.ipynb` |
| 实践 (20min) | 运行 `27_multi_token_prediction.ipynb`，对比单/多 Token 预测效率 | `27_multi_token_prediction.ipynb` |
| 笔记 (10min) | 记录长上下文优化策略和多 Token 预测的关键洞察 | — |

**✅ 检查点**：能解释为什么 LLM 对中间位置的信息处理较差

### Day 27（周六）：理论深潜 — 复杂性与超级智能

| 时段 | 内容 | 文件 |
|------|------|------|
| 理论 (10min) | 复杂性动力学、热力学不可逆性、Coffee Automaton 核心思想 | — |
| 实践 (15min) | 浏览 `01_complexity_dynamics.ipynb`，理解熵增与复杂度的关系 | `01_complexity_dynamics.ipynb` |
| 实践 (15min) | 浏览 `24_machine_super_intelligence.ipynb` 的 Section 1-3（AIXI、通用智能度量） | `24_machine_super_intelligence.ipynb` |
| 笔记 (10min) | 画出从 Kolmogorov → Solomonoff → AIXI 的理论链条 | — |
| 关联 (10min) | 回顾全部 30 篇论文的知识图谱，整理论文间的技术传承关系 | — |

**✅ 检查点**：能描述 Solomonoff 归纳和 AIXI 的基本思想

### Day 28（周日）：总自测 + 知识整合

| 时段 | 内容 |
|------|------|
| 复习 (15min) | 最终间隔复习：Day 1, 4, 8, 11, 15, 19 的核心内容 |
| 自测 (45min) | 完成下方终极自测题 |

#### 📋 第四周终极自测题

1. **知识图谱题**：画出一张包含至少 10 篇论文的知识关联图，标注技术传承关系（如：Bahdanau Attention → Transformer → Pointer Networks → RAG）
2. **系统设计题**：设计一个基于 RAG 的问答系统，画出完整架构图，标注使用了本课程中哪些技术（DPR、Transformer、Attention 等）
3. **深度分析题**：从"记忆"的角度，写一篇 500 字的短文，比较 LSTM、NTM、Transformer、RAG 四种模型如何存储和访问知识
4. **缩放题**：如果你有一笔固定的训练预算（GPU 小时数），根据 Scaling Laws 你会如何分配模型大小和数据量？给出你的推理过程
5. **编程题**：不看代码，手写一个简化版的 RAG 推理流程的 Python 伪代码（包括 Retrieve + Generate 两个步骤）
6. **理论题**：用你自己的话解释"为什么压缩是智能的本质"，引用至少 3 个本课程中学到的概念（MDL、Kolmogorov 复杂度、Solomonoff 归纳等）

**🎯 及格标准**：6 题答对 4 题以上

### 📊 第四周可衡量成果

- [ ] 能画出 30 篇论文的知识关联图
- [ ] 能设计一个完整的 RAG 系统架构
- [ ] 能用缩放律指导训练预算分配
- [ ] 能解释从基础模型到现代 LLM 的完整技术演进
- [ ] 完成终极自测题得分 ≥ 66%
- [ ] 能选择任意一个主题向他人做 10 分钟的技术讲解

---

## 📖 推荐学习资源

### 必读资源

| 资源 | 类型 | 对应内容 | 链接 |
|------|------|---------|------|
| Andrej Karpathy's Blog | 博客 | RNN、Char RNN (Paper 2) | [karpathy.github.io](http://karpathy.github.io/) |
| Christopher Olah's Blog | 博客 | LSTM (Paper 3)、Attention | [colah.github.io](https://colah.github.io/) |
| The Annotated Transformer | 教程 | Transformer (Paper 13) | [nlp.seas.harvard.edu](http://nlp.seas.harvard.edu/annotated-transformer/) |
| Ilya Sutskever's Reading List | 论文集 | 全部 30 篇原始论文 | [GitHub](https://github.com/dzyim/ilya-sutskever-recommended-reading) |
| Aman's AI Journal Primers | 教程 | 所有论文速览 | [aman.ai](https://aman.ai/primers/ai/top-30-papers/) |

### 视频课程

| 课程 | 对应周次 | 适用场景 |
|------|---------|---------|
| Stanford CS231n (YouTube) | 第 1-2 周 | CNN 和视觉基础 |
| Stanford CS224n (YouTube) | 第 2-3 周 | NLP 和序列模型 |
| MIT 6.S191 | 全程 | 快速概览深度学习 |
| 3Blue1Brown 神经网络系列 | 第 1 周 | 直觉建立 |

### 进阶资源

| 资源 | 适用时机 |
|------|---------|
| Lilian Weng's Blog (lilianweng.github.io) | 第 3-4 周，综合理解各主题 |
| Jay Alammar's Blog (jalammar.github.io) | 第 2 周，Transformer 可视化理解 |
| Distill.pub 文章 | 第 2-3 周，交互式理解注意力、特征可视化 |
| Chinchilla Paper (Hoffmann et al. 2022) | 第 4 周，Scaling Laws 深入 |

---

## ⚠️ 风险点与瓶颈分析

### 1. 数学门槛风险

| 风险 | 出现时机 | 影响 | 缓解方案 |
|------|---------|------|---------|
| VAE 的变分推断公式 | 第 3 周 Day 15 | 可能卡在 ELBO 推导上 | 先理解直觉（编码-采样-解码），再补公式推导 |
| CTC 的前向算法 | 第 3 周 Day 19 | 动态规划 + 概率推导较复杂 | 先画格状图建立直觉，再看代码实现 |
| Kolmogorov 复杂度的不可计算性 | 第 3 周 Day 20 | 纯理论，难以实践验证 | 聚焦"压缩 ≈ 理解"的应用直觉 |

### 2. 时间管理风险

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| 大型 Notebook 时间超标 | Paper 18（1100行）、19（2500行）、24（2000行）、26（2400行）耗时可能超过 1 小时 | 这些内容已安排浏览而非精读；只看核心 Section |
| 复习时间累积 | 随着间隔复习的叠加，后期复习负担增加 | 复习只做"闭卷回忆 + 快速验证"，严控 10 分钟内 |
| 自测题准备 | 周日自测可能需要超过 40 分钟 | 允许拆分到两个时段完成 |

### 3. 理解深度风险

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| 流于表面 | 每天只有 1 小时，可能导致"走马观花" | 每天设有明确检查点；每周自测保证深度 |
| 跨论文联系缺失 | 单独学习每篇论文，缺少整体视角 | 每天有 5 分钟"关联思考"环节；Day 27 做全图谱整理 |
| 理论与实践脱节 | 理论学了但不知道怎么用 | 编程题要求不看代码手写；第 4 周有系统设计题 |

---

## 📊 计划可行性评估

### 总体评估：⭐⭐⭐⭐ (4/5) — 有挑战但可执行

#### ✅ 优势

1. **内容自洽**：所有 Notebook 自包含，无需额外数据下载，可立即运行
2. **难度递增合理**：第 1 周（RNN/LSTM/CNN）→ 第 2 周（Attention/Transformer）→ 第 3 周（VAE/NTM/理论）→ 第 4 周（RAG/缩放律/综合），符合认知负荷递增原则
3. **覆盖完整**：4 周覆盖 30 篇论文中的 28 篇（Paper 19 Coffee Automaton 和 Paper 26 CS231n 仅安排浏览），覆盖率 93%
4. **复习机制有效**：基于间隔重复的复习时间点有科学依据

#### ⚠️ 潜在问题

1. **单日强度偏高**：第 3 周的 VAE + NTM + CTC 连续三天高难度，可能导致认知疲劳
2. **第 4 周综合跳跃**：从理论突然跳到前沿应用（RAG），衔接可能不够平滑
3. **大型 Notebook 时间估算**：部分 Notebook 内容量大，30 分钟可能不够完整运行和理解

#### 📈 强度曲线分析

```
认知负荷
  ▲
  │          ★ Day 15 (VAE)
  │    ★ Day 10     ★ Day 18 (NTM)
  │   (Transformer)      ★ Day 20 (Info Theory)
  │  ★ Day 8              ★ Day 22 (Scaling)
  │  (Attention)                ★ Day 25 (RAG)
  │ ★ Day 6                         ★ Day 27 (综合)
  │ (ResNet)
  │★ Day 2
  │(LSTM)
  │★ Day 1
  │(RNN)
  └──────────────────────────────────────────────► 天
    1  2  3  4  5  6  7  8  10 15 18 20 22 25 27
```

强度曲线整体呈递增趋势，每周内部有高低交替（周三安排复习日缓冲、周六安排较轻内容），避免持续高强度。

### 优化建议

1. **弹性缓冲日**：如果某天进度落后，可将周六的双 Notebook 日拆成两天，用周日上午补进度
2. **难度分散**：如果第 3 周感觉过难，可将 Day 20（MDL + Kolmogorov）提前到第 2 周的 Day 13 替换膨胀卷积
3. **深度优先策略**：如果时间不足，优先保证以下 10 篇核心论文的深度理解：
   - Paper 2 (RNN), 3 (LSTM), 13 (Transformer), 14 (Attention), 17 (VAE)
   - Paper 10 (ResNet), 20 (NTM), 22 (Scaling Laws), 28 (DPR), 29 (RAG)
4. **CS231n 单独安排**：Paper 26 (CS231n) 内容量大（2400 行），如果对 CNN 基础需要强化，可替换第 1 周中的 Day 5 (AlexNet)，用 CS231n 做更系统的学习
5. **社区互动**：建议在学习过程中将学习笔记以 Issue 或 Discussion 形式发布到 GitHub，形成外部反馈循环

---

## 🗺️ 全局知识地图

学完 4 周后，你将建立以下知识体系：

```
                    ┌─────────────────────────────────┐
                    │   Week 4: 综合应用与前沿         │
                    │   Scaling Laws · GPipe · RAG     │
                    │   DPR · Multi-token · 超级智能   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │   Week 3: 高级架构与理论         │
                    │   VAE · NTM · Relational RNN     │
                    │   CTC · MDL · Kolmogorov         │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │   Week 2: 注意力与现代架构       │
                    │   Bahdanau · Transformer · GNN   │
                    │   Pointer Net · Sets · ResNet v2 │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │   Week 1: 基础                   │
                    │   RNN · LSTM · Dropout · Pruning │
                    │   CNN · ResNet                    │
                    └─────────────────────────────────┘
```

**核心技术链条**：
- **序列建模**：RNN → LSTM → Attention → Transformer
- **视觉理解**：CNN → ResNet → Pre-activation → Dilated Conv
- **记忆机制**：LSTM → NTM → Relational RNN → Transformer KV Cache
- **检索与生成**：Dense Retrieval → RAG → Long Context
- **理论基础**：MDL → Kolmogorov → Solomonoff → AIXI

---

> **"If you really learn all of these, you'll know 90% of what matters today."** — Ilya Sutskever
>
> 祝学习顺利！🚀
