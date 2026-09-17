# NeuroRoute 全流程实验总报告

> 状态日期：2026-06-23  
> 项目：UMBRAE + NeuroRoute  
> 受试者：S1、S2、S5、S7；下游 caption 实验目前仅完成 S1

## 1. 总体研究问题

原始 UMBRAE 将整个 `nsdgeneral.npy` fMRI 向量编码成 256 个视觉
token，并对齐 CLIP 最终语义表示后送入 Shikra。NeuroRoute 试图回答：

1. 能否用真实脑区映射将 fMRI 拆成稳定的 ROI token？
2. CLIP 多层特征是否比单一最终层提供更好的脑表征监督？
3. 可学习 ROI-to-layer routing 是否优于 uniform、hard 和单层路由？
4. 对齐优势是否转化为 retrieval 和 caption 优势？
5. ROI token 应替换 UMBRAE，还是作为 UMBRAE 的补充？

截至目前，最简洁的结论是：

> 真实 ROI tokenization 可行；soft multi-layer routing 稳定改善
> Stage-1 alignment，但没有学出明确的 ROI-specific 层级。检索收益具有
> 任务依赖性。在严格 Shikra caption 实验中，ROI 多层 augmentation
> 优于 UMBRAE-only，但 uniform coverage 强于 learned soft routing；
> NeuroRoute 更适合作为 UMBRAE 的补充，而不是替代。

## 2. 实验路线总览

| 阶段 | 为什么做 | 目的 | 主要效果 | 阶段结论 |
|---|---|---|---|---|
| 数据/ROI 对齐 | UMBRAE 的 `nsdgeneral.npy` 没有直接携带 ROI 信息 | 建立真实 subject-specific ROI index | S1/S2/S5/S7 均完成 voxel-order 验证 | 8 ROI 可以安全用于真实 fMRI |
| CLIP Multi-Layer Bank | 原方法主要使用最终层 | 提供 L4/8/12/16/20/24 多层监督 | 模块和 shape 测试通过 | 可在不改变默认 UMBRAE 路径下提取多层特征 |
| ROI Tokenizer | 整体脑向量无法显式建模脑区 | 将 fMRI 变成 8 个 ROI token | 真实映射测试 `[1,3,V] -> [1,8,32]` 通过 | ROI-wise 表征管线成立 |
| Stage-1 soft routing | 检验学习式多层混合是否有效 | 对齐 ROI token 与 routed CLIP target | 四受试者 alignment 均优于 uniform/L24 | 学习式多层混合对 alignment 有效 |
| Collapse/正则化 | soft routing 偏向 L24 | 检查 temperature/balance 能否产生 ROI 分层 | 可降低最大权重，但无 ROI top-layer 分化 | collapse 缓解不等于脑区层级恢复 |
| BrainToCLIPProjector | raw ROI token 与 CLIP 分布不同 | 解耦脑区聚合和 CLIP 空间校准 | alignment loss 大幅下降且训练更稳定 | projector 是必要的表示桥梁 |
| Stage-1 retrieval | alignment loss 不保证实例判别 | 检查脑到图像检索 | contrastive 提升 B2I R@5/R@10，但损害 alignment | 收益是任务权衡，不是全面提升 |
| Generic-prefix Stage-2 | 快速验证 ROI token 下游价值 | 在通用 `inputs_embeds` 下做 caption ablation | concat-soft 综合优于 L24-only | ROI token 有下游潜力，但不是严格 UMBRAE |
| Strict UMBRAE/Shikra 集成 | generic prefix 与论文协议不公平 | 使用 256 个 `<im_patch>` 原位替换 | uniform ROI augmentation caption 最强 | ROI augmentation 有效，soft 不如 uniform |
| Structured routing | uniform 胜 soft，怀疑 coverage 更重要 | uniform 基础上加入有限 soft 偏移 | alpha/tau 均未超过 uniform | 当前任务更依赖稳定覆盖而非 learned specialization |

---

## 3. 数据与真实 ROI 映射实验

### 3.1 为什么先做 wholebrain/ROI 对齐

NSD ROI atlas 位于 `func1pt8mm` 三维体空间，而 UMBRAE 模型读取的是
扁平 `nsdgeneral.npy`。三维 atlas flat index 不能直接当作
`nsdgeneral` vector index，否则会将错误 voxel 分配给 ROI。

因此先验证：

```text
wholebrain_3d.npy spatial shape
    == official ROI NIfTI shape

wholebrain[nsdgeneral_mask]
    ≈ sample.nsdgeneral.npy
```

实际验证中，空间 shape 一致，数值差异最大约 `0.001953125`，符合
float16 存储误差。

### 3.2 ROI index 转换目的

进行了两级转换：

```text
ROI NIfTI label
  -> wholebrain x/y/z 和 C-order flat index
  -> nsdgeneral.npy vector position
```

并用真实样本同时从：

```text
wholebrain_3d.npy
nsdgeneral.npy
```

提取同一 ROI，relaxed allclose 全部通过。只有通过真实样本验证的 subject
才设置：

```json
"voxel_order_verified": true
```

### 3.3 最终 NeuroRoute-v1 ROI set

最终使用：

```text
V1, V2, V3, hV4, FFA, EBA, PPA, OPA
```

ROI voxel 数：

| Subject | V1 | V2 | V3 | hV4 | FFA | EBA | PPA | OPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1350 | 1433 | 1187 | 687 | 687 | 2238 | 720 | 1532 |
| S2 | 1102 | 1075 | 1097 | 483 | 868 | 2351 | 755 | 1322 |
| S5 | 1113 | 1081 | 925 | 542 | 763 | 2869 | 975 | 1162 |
| S7 | 1142 | 986 | 726 | 397 | 343 | 2294 | 652 | 1078 |

### 3.4 为什么没有 RSC

官方 `floc-places:RSC` 在 wholebrain 空间存在，但对 S1/S2/S5/S7：

```text
RSC ∩ nsdgeneral = 0 voxel
```

因此 RSC 被记录为 unavailable/excluded，而不是创建空 ROI 或伪造 index。

### 3.5 效果与结论

- 四个 subject 的真实 ROI mapping 均完成。
- ROI tokenizer 真实测试通过：

```text
[B,3,V] -> [B,8,D]
```

- 这是后续所有 ROI 实验的基础。
- 当前结论只适用于 UMBRAE 的 `nsdgeneral` 输入空间；若要支持 RSC，
  必须扩展到 wholebrain 或更大的 mask。
- 另外生成了 S1/S2/S5/S7 的 wholebrain ROI 三视图/体空间 QC
  visualization，用于检查分区坐标和 label 空间位置；它是数据 sanity
  check，不是模型性能实验。

---

## 4. 基础模块实验

### 4.1 CLIP Multi-Layer Bank

**为什么做：** 检验脑表征是否能从 CLIP 中间层获得最终层之外的信息。

选择层：

```text
L4, L8, L12, L16, L20, L24
```

输出：

```text
layer_tokens  [B,6,256,D]
pooled_tokens [B,6,D]
```

CLIP 默认冻结，每层有独立 projection head。模块 smoke test 通过，且默认
UMBRAE 行为不受影响。

### 4.2 ROITokenizer

**为什么做：** 原始整体脑表征无法保留显式脑区身份。

流程：

```text
ROI voxel subset
 -> shared/ROI-specific MLP
 -> ROI identity embedding
 -> [B,8,D]
```

当前真实实验主要使用 shared MLP 和 ROI identity embedding。

### 4.3 ROILayerRouter

**为什么做：** 让每个 ROI 自适应选择 CLIP 多层目标。

```text
A[r,l] = softmax(Q(z_r)·K(v_l) / sqrt(d) / tau)
routed_target_r = sum_l A[r,l] v_l
```

测试验证：

- routing weights 对 layer 维求和为 1；
- top-k、CPU、CUDA 正常；
- entropy、balance、smoothness loss 有限；
- 不使用对 softmax 无效的 L1 penalty。

这些测试证明模块正确，但不等同于证明神经科学层级假设。

---

## 5. Stage-1 初步 routing 与 collapse 实验

### 5.1 初始 soft/uniform/random/hard

**为什么做：** 先判断 learnable soft routing 是否优于固定规则。

S1 早期实验没有 projector：

| 方法 | Best val alignment loss |
|---|---:|
| Soft | **1.16025** |
| Uniform | 1.25154 |
| Random | 1.26874 |
| Hard | 1.29661 |

**效果：** soft 最好，但 routing 强烈偏向 L24，平均最大权重约 0.726。

**结论：**

- learnable mixture 比固定 baseline 更适合当前 alignment loss；
- 但它主要学到深层偏置，不能解释为 ROI-specific hierarchy。

### 5.2 temperature/balance anti-collapse

**为什么做：** 检查 L24 dominance 是否只是温度过低或缺少 balance。

| 设置 | Val loss | Routing entropy | Mean max weight |
|---|---:|---:|---:|
| Soft | 1.16025 | 0.893 | 0.726 |
| Balance 0.01 | 1.16568 | 1.044 | 0.686 |
| Balance 0.05 | 1.18479 | 1.391 | 0.535 |
| Balance 0.10 | 1.19998 | 1.551 | 0.438 |
| Tau 2 | **1.15143** | 0.859 | 0.761 |
| Tau 2 + balance 0.05 | 1.18360 | 1.373 | 0.550 |

**效果：**

- balance 能提高 entropy、降低最大层权重；
- 但 alignment loss 随 balance 增大而变差；
- temperature sweep 没有产生稳定 ROI-specific top layer。

**结论：**

> 降低 collapse 可以改善 routing 分布外观，但没有恢复可信的脑区层级，
> 并可能损害主任务。

---

## 6. BrainToCLIPProjector 实验

### 6.1 为什么增加 projector

raw ROI token 同时承担：

1. ROI voxel 聚合；
2. ROI identity 表达；
3. 匹配 CLIP embedding 的尺度与坐标分布。

直接对齐会让 tokenizer 承担过多职责。因此增加：

```text
raw ROI token
 -> BrainToCLIPProjector
 -> CLIP-aligned ROI token
```

router query 仍来自 raw ROI token，以保留可解释性。

### 6.2 S1 projector baseline

500-step 对比：

| 方法 | Best val loss |
|---|---:|
| Soft + MLP projector | **0.11343** |
| Soft + balance 0.05 | 0.12102 |
| Uniform + projector | 0.14025 |
| Random + projector | 0.15095 |
| Hard + projector | 0.16190 |
| Single L24 + projector | 0.14770 |

单层扫描：

| L4 | L8 | L12 | L16 | L20 | L24 |
|---:|---:|---:|---:|---:|---:|
| 0.1895 | 0.2647 | 0.2415 | 0.1877 | 0.2141 | **0.1477** |

**效果：**

- projector 使 alignment loss 从约 1.x 降至约 0.1；
- soft 仍优于所有固定 baseline；
- 所有 ROI 的 top-1 仍然偏向 L24。

**结论：**

> projector 解决了表示分布不匹配；soft 的收益不是单纯尺度问题。但
> projector 也没有产生 ROI-specific layer hierarchy。

### 6.3 Longer-training routing dynamics

**为什么做：** 检查 500-step 结果是否只是训练不足，继续训练是否会让
不同 ROI 逐渐选择不同层。

S1 soft + projector 延长到 10 epochs 后：

- validation alignment 继续稳定；
- routing 继续收敛到 L24-biased multi-layer mixture；
- 8 个 ROI 的 top-1 仍然都是 L24；
- early/high-level ROI expected depth 没有形成可解释分离。

**结论：** “没有 ROI-specific hierarchy”不是单纯由训练太短造成的。

---

## 7. Stage-1 四受试者 alignment 实验

### 7.1 为什么跨 subject 验证

S1 单 subject 的优势可能是随机性或 subject-specific artifact。于是固定：

- 相同 8 ROI；
- 相同 CLIP layers；
- 相同 projector；
- 相同训练长度、batch size 和学习率；
- 对 S1/S2/S5/S7 分别独立训练。

### 7.2 结果

Best validation alignment loss：

| Subject | Soft | Uniform | Single-L24 | Soft vs uniform | Soft vs L24 |
|---|---:|---:|---:|---:|---:|
| S1 | **0.103179** | 0.129467 | 0.141285 | +20.30% | +26.97% |
| S2 | **0.102983** | 0.128507 | 0.140701 | +19.86% | +26.81% |
| S5 | **0.101401** | 0.125884 | 0.139145 | +19.45% | +27.13% |
| S7 | **0.103666** | 0.130385 | 0.141405 | +20.49% | +26.69% |

平均：

| 方法 | Mean best val loss |
|---|---:|
| Soft | **0.102807** |
| Uniform | 0.128561 |
| Single-L24 | 0.140634 |

### 7.3 Routing 结构

| 方法 | Mean L24 usage | Early ROI depth | High-level ROI depth | Gap |
|---|---:|---:|---:|---:|
| Soft | 0.3460 | 17.504 | 17.533 | +0.029 |
| Uniform | 0.1667 | 14.000 | 14.000 | 0 |
| Single-L24 | 1.0000 | 24.000 | 24.000 | 0 |

### 7.4 效果与结论

**支持：**

- soft multi-layer routing 对 alignment 的优势跨四个 subject 稳定；
- 非 L24 层确实有辅助信息，因为 soft 优于 single-L24。

**不支持：**

- early visual ROI 对浅层、FFA/PPA 对深层的明确分工；
- routing 恢复神经解剖层级。

正确表述是：

> soft routing 学到 L24-biased adaptive multi-layer mixture，而不是
> ROI-specific hierarchy。

---

## 8. Stage-1 retrieval 实验

### 8.1 为什么需要 retrieval

MSE + cosine alignment 只说明匹配目标空间，不保证不同样本可被区分。
因此增加 fMRI-to-image 和 image-to-fMRI retrieval。

协议：

```text
brain query = fMRI -> ROI tokens -> projector -> mean ROI
image gallery = frozen CLIP L24
similarity = normalized brain @ normalized image.T
```

按 `coco73k` ID 建立 multi-positive mask，不默认只有对角线为正样本。当前
300-sample shard 恰好含 300 个 unique ID，所以每个 query 实际只有一个
positive。

### 8.2 失败的 attention+routed 实验

**为什么做：** 希望 attention pooling 和 routed image target 提高判别性。

**效果：** mean/attention × L24/routed 四种组合均接近 300-way random。

**诊断：**

- attention pooler 权重存在，不是“未加载随机 head”；
- checkpoint 只训练 1 epoch、10 steps；
- routed gallery target 也不稳定。

**结论：** 这是 undertrained smoke checkpoint，不用于正式结论；
attention+routed 不作为默认协议。

### 8.3 Conservative contrastive sweep

改用更保守设置：

```text
brain pooling = mean
image target = L24
InfoNCE temperature = 0.07
weight = 0.01 / 0.03 / 0.05
```

S1 B2I：

| 方法 | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft alignment-only | 1.00% | 3.33% | 5.33% | 120.5 | 127.53 |
| Uniform alignment-only | 0.33% | 3.00% | 5.33% | 124.5 | 132.79 |
| Single-L24 | 0.67% | 4.00% | 6.00% | 108.0 | 119.46 |
| Contrastive 0.01 | 1.00% | 3.67% | 6.67% | 112.0 | 122.36 |
| Contrastive 0.03 | 1.00% | 5.00% | 7.33% | 98.5 | 115.14 |
| Contrastive 0.05 | 1.00% | **5.00%** | **8.00%** | **96.5** | **112.68** |

### 8.4 四受试者平均 retrieval

B2I：

| 方法 | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft | 0.92% | 3.17% | 5.75% | 120.88 | 129.42 |
| Uniform | 0.42% | 2.75% | 5.08% | 124.38 | 133.03 |
| Single-L24 | **1.42%** | 4.67% | 7.58% | 109.75 | 121.74 |
| Soft contrastive 0.05 | 1.08% | **4.83%** | **9.92%** | **94.63** | **112.08** |

I2B：

| 方法 | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft | 2.67% | 12.50% | 21.75% | 39.38 | 61.32 |
| Uniform | 2.25% | 8.83% | 15.17% | 56.63 | 77.78 |
| Single-L24 | **6.33%** | **18.42%** | **31.00%** | **28.00** | **45.82** |
| Soft contrastive 0.05 | 4.67% | 17.75% | 28.83% | 29.13 | 48.24 |

Contrastive 0.05 同时令 alignment component 下降约 5.5%–7.7%。

### 8.5 Retrieval 结论

- alignment-only soft 略优于 uniform retrieval；
- soft 没有全面超过 single-L24；
- contrastive 0.05 提供最佳平均 B2I R@10 和 rank；
- single-L24 保持最佳 B2I R@1 和多数 I2B 指标。

因此：

> alignment 与 retrieval 存在任务权衡；不能用 alignment loss 宣称检索
> 全面提升，也不能宣称 soft routing 对所有任务都最优。

---

## 9. Caption ID 数据准备实验

### 9.1 为什么最初 coverage 很低

早期直接把 `coco73k.npy` 当作 raw COCO image ID，导致：

- S1 val 只有 114/300 匹配；
- S1 train 前 100 个样本 0/100 匹配。

### 9.2 ID 审计结果

确认：

```text
coco73k.npy = zero-based NSD nsdId
nsdId -> nsd_stim_info_merged.csv -> raw COCO cocoId
cocoId -> COCO 2017 captions
```

最终映射：

```text
stage2_outputs/caption_mapping/coco73k_captions.json
```

覆盖：

| Subject | Train | Validation |
|---|---:|---:|
| S1 | 8559/8559 | 300/300 |
| S2 | 8559/8559 | 300/300 |
| S5 | 8181/8181 | 300/300 |
| S7 | 8189/8189 | 300/300 |

总计 34,688 unique local IDs，100% 匹配。

**结论：** caption benchmark 的数据 blocker 已解决；旧
`fmri_cococap.json` 不是完整训练映射。

---

## 10. Generic-prefix Stage-2 caption 实验

### 10.1 为什么先做 generic prefix

严格 Shikra bridge 尚未实现时，先用通用 `inputs_embeds` prefix 快速判断
ROI token 是否有下游信号。该实验不是论文级 UMBRAE 复现。

设计：

- `l24_only`：全局语义路径；
- `routed_only`：8 个 ROI token；
- `concat_soft`：L24 + soft ROI；
- `concat_uniform`：L24 + uniform ROI；
- `concat_single_l24`：L24 + L24-aligned ROI。

所有 concat 使用同一个 global L24 checkpoint。

### 10.2 S1 三 epoch 结果

| 方法 | Val loss ↓ | BLEU-4 ↑ | CIDEr ↑ | ROUGE-L ↑ |
|---|---:|---:|---:|---:|
| L24 only | 2.2089 | 0.0270 | 0.1828 | 0.2205 |
| Routed only | 2.1860 | **0.0359** | **0.2123** | 0.2213 |
| Concat soft | **2.1543** | 0.0320 | 0.2012 | **0.2264** |
| Concat uniform | 2.1831 | 0.0316 | 0.1755 | 0.2121 |
| Concat single-L24 | 2.2038 | 0.0351 | 0.1952 | 0.2231 |

Concat-soft val loss 相对：

- L24-only：改善 2.47%；
- concat-uniform：改善 1.32%；
- concat-single-L24：改善 2.24%。

### 10.3 效果与结论

- ROI token 在 generic prefix 下有下游价值；
- concat-soft 综合最平衡；
- routed-only 的 BLEU-4/CIDEr 最好，说明 ROI token 并非完全不可独立使用；
- soft 与 single-L24 的高级 BLEU 指标有交叉，证据不是全指标一致。

限制：

- 不是 `<im_patch>` bridge；
- S1 only；
- 每张图当前评测只使用一个参考 caption；
- 不可与论文 UMBRAE 数值直接比较。

---

## 11. Strict UMBRAE/Shikra 集成实验

### 11.1 为什么必须重新集成

Generic prefix 改变了原始视觉 token 协议。为了更公平，审计原始代码并确认：

```text
BrainX/BrainXS -> [B,256,1024]
mm_projector   -> [B,256,4096]
prompt         -> <im_start> + 256*<im_patch> + <im_end>
```

于是实现严格原位替换：

```text
256 UMBRAE tokens + 8 ROI tokens
 -> 256-query Perceiver
 -> [B,256,4096]
 -> 替换 256 个 <im_patch>
```

UMBRAE、NeuroRoute、Shikra 全冻结，只训练 fusion adapter。

### 11.2 为什么比较五种模式

| 模式 | 要回答的问题 |
|---|---|
| UMBRAE-only | 原全局路径基线 |
| NeuroRoute-only | ROI token 能否替代 UMBRAE |
| UMBRAE + soft | 主 NeuroRoute 方法是否改善 UMBRAE |
| UMBRAE + uniform | 提升来自 learnable routing 还是多层覆盖 |
| UMBRAE + single-L24 | 多层 ROI 是否优于重复最终层语义 |

### 11.3 S1 结果

| 方法 | Best val ↓ | Final val ↓ | BLEU-4 ↑ | ROUGE-L ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|---:|
| UMBRAE-only | **1.884107** | 1.916646 | 0.071218 | 0.294891 | 0.594173 |
| NeuroRoute-only | 2.175133 | 2.175133 | 0.041993 | 0.230652 | 0.246029 |
| UMBRAE + soft | 1.909896 | 1.909896 | 0.084187 | 0.304803 | 0.713035 |
| UMBRAE + uniform | 1.908940 | **1.908940** | **0.091184** | **0.307187** | **0.758230** |
| UMBRAE + single-L24 | 1.922040 | 1.922040 | 0.072228 | 0.284027 | 0.537156 |

Soft 相对 UMBRAE-only：

- BLEU-4：+18.21%；
- CIDEr：+20.00%；
- ROUGE-L：+3.36%；
- final val loss：改善 0.35%；
- best val loss：差 1.37%。

Soft 相对 single-L24：

- BLEU-4：+16.56%；
- CIDEr：+32.74%；
- ROUGE-L：+7.31%。

### 11.4 Strict caption 结论

- ROI augmentation 改善最终 caption metrics；
- NeuroRoute-only 明显更弱，因此 ROI token 应作为 UMBRAE 的补充；
- multi-layer ROI 优于 single-L24 ROI；
- uniform 全指标优于 soft，learnable routing 不是当前 caption 最佳策略；
- UMBRAE-only 仍有最低 best val loss，说明 caption metric 与 language-model
  validation loss 并不完全一致。

### 11.5 与论文 UMBRAE 的关系

当前 `UMBRAE-only` 是：

```text
frozen BrainX + frozen Shikra + 3-epoch adapter tuning
```

不是论文的零训练原始推理复现。另有以下差异：

- 使用 8559 train / 300 val，而原始训练代码将 300 val 并入 train，并在
  982 test 上验证；
- 使用 `last.pth`，不一定是论文 best checkpoint；
- 当前单参考 caption，而 COCO/论文可能使用多参考；
- prompt、生成后处理和正式 BrainHub 评测协议未完全一致。

因此 strict bridge 更公平，但仍不能直接宣称复现论文最终指标。

---

## 12. Structured routing 改进实验

### 12.1 为什么做

Strict caption 中：

```text
uniform > soft > single-L24
```

假设 soft 的深层偏置与 UMBRAE 全局语义 token 冗余，而 uniform 因稳定覆盖
六层而更强。于是测试：

```text
A = (1-alpha) A_uniform + alpha A_soft
```

以及：

```text
A_temp = softmax(log(A_soft + eps) / tau)
```

只新增 alpha 0.1/0.3/0.5 和 tau 2.0，旧 baseline 没有重跑。

### 12.2 结果

| 方法 | Final val ↓ | BLEU-4 ↑ | ROUGE-L ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|
| Uniform | 1.908940 | **0.091184** | **0.307187** | **0.758230** |
| Soft | 1.909896 | 0.084187 | 0.304803 | 0.713035 |
| Alpha 0.1 | 1.910761 | 0.077578 | 0.292739 | 0.604175 |
| Alpha 0.3 | **1.908892** | 0.077615 | 0.295642 | 0.615365 |
| Alpha 0.5 | 1.915305 | 0.074344 | 0.287580 | 0.578109 |
| Temperature 2.0 | 1.918370 | 0.074060 | 0.290924 | 0.594332 |

Alpha 0.3 虽然 val loss 比 uniform 低 0.0025%，但：

- BLEU-4 低 14.88%；
- CIDEr 低 18.84%；
- ROUGE-L 低 3.76%。

### 12.3 Routing diagnostics

| Routing | Entropy | Coverage | ROI diversity | Early depth | High depth | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 1.7918 | 1.0000 | 0.0000 | 14.0000 | 14.0000 | 0.0000 |
| Alpha 0.1 | 1.7883 | 0.9966 | 0.0017 | 14.3427 | 14.3417 | -0.0011 |
| Alpha 0.3 | 1.7611 | 0.9698 | 0.0050 | 15.0282 | 15.0250 | -0.0032 |
| Alpha 0.5 | 1.7056 | 0.9176 | 0.0083 | 15.7136 | 15.7083 | -0.0053 |
| Tau 2.0 | 1.6584 | 0.8752 | 0.0090 | 16.1161 | 16.1140 | -0.0021 |
| Soft | 1.4124 | 0.6845 | 0.0165 | 17.4273 | 17.4167 | -0.0106 |

### 12.4 Structured routing 结论

- alpha/tau 均未超过 uniform；
- alpha 越大，expected depth 越深、coverage 越低、caption 趋势越差；
- tau 2 缓解 soft collapse，但不足以改善 caption；
- ROI diversity 很低，early/high gap 接近 0。

因此：

> S1 caption 更依赖稳定、广泛的多层覆盖，而不是当前学到的 ROI-layer
> specialization。

---

## 13. 泄漏与协议安全检查

所有正式 retrieval/caption 运行遵循：

```text
fMRI query/prefix 只来自 fMRI
image CLIP feature 不进入 brain query
oracle_image_token_mode = false
```

Strict UMBRAE–NeuroRoute 运行还记录：

```text
bridge_type = shikra_patch
uses_image_clip_tokens_at_eval = false
```

Stage-1 中 image CLIP features 仅作为冻结监督 target；它们不会在
validation/test 时被用于构造 fMRI token。

---

## 14. 当前可支持的研究结论

### 14.1 可以支持

1. 已建立 S1/S2/S5/S7 的真实、voxel-order-verified 8-ROI mapping。
2. ROI-wise fMRI tokenization 能兼容 UMBRAE `nsdgeneral.npy`。
3. projector-enabled soft multi-layer routing 在四受试者 Stage-1
   alignment 上稳定优于 uniform 和 single-L24。
4. soft routing 学到的是 L24-biased multi-layer mixture。
5. conservative contrastive loss 改善 B2I R@5/R@10 和 rank，但牺牲
   alignment。
6. S1 strict-Shikra caption 中，ROI multi-layer augmentation 改善
   UMBRAE-only 的最终 caption metrics。
7. NeuroRoute-only 弱于融合方法，ROI token 更适合作为补充。
8. 当前 S1 caption 中 uniform multi-layer coverage 是最强 routing
   baseline。

### 14.2 不能支持

1. 不能声称存在清晰的 ROI-specific CLIP hierarchy。
2. 不能声称 soft routing 在所有任务上优于 uniform 或 single-L24。
3. 不能声称 retrieval 全面提升。
4. 不能声称跨受试者 caption 提升，因为 strict caption 尚只跑 S1。
5. 不能声称论文级 UMBRAE 复现或 SOTA。
6. 不能将当前单参考、300-val caption 数值直接与论文表格相减作为公平
   对比。

---

## 15. 当前整体判断

所有实验合起来形成了一个较清楚的图景：

```text
真实 ROI 信息
  -> 对齐阶段有稳定价值
  -> 下游也能提供补充信息
  -> 但当前 learned ROI-to-layer 权重缺少 ROI 差异
  -> caption 最需要的是多层覆盖，而不是更深的 soft 偏置
```

因此当前最稳妥的 NeuroRoute-v1 方法不是“用路由替代 UMBRAE”，而是：

```text
保留 UMBRAE 256 global visual tokens
+ 加入真实 ROI-derived multi-layer tokens
+ 用稳定 coverage 的融合方式注入 Shikra
```

目前 S1 caption 的最佳实现是 `UMBRAE + uniform ROI multi-layer
augmentation`。Soft routing 的主要科学价值仍体现在 Stage-1 alignment，
而不是已被证明的脑区层级或最终 caption 最优性。

### 15.1 为什么 generic-prefix 与 strict-Shikra 得到不同 routing 排名

Generic-prefix 实验中 `concat-soft` 综合优于 `concat-uniform`，而 strict
UMBRAE/Shikra 中 uniform 最强。这不是直接矛盾，因为两者的信息结构不同：

- generic-prefix 只有一个 global L24 token 加 8 个 ROI token；
- strict 集成保留了原始 UMBRAE 的 256 个全局视觉 token；
- 在后者中，soft routing 的深层语义偏置更可能与 UMBRAE 全局 token
  冗余，而 uniform 能补充更广的中间层覆盖。

因此 routing 优劣依赖下游 bridge 和已有 global path，不能脱离模型上下文
单独宣称某种 routing 永远最优。

---

## 16. 原始 UMBRAE 训练设置（对照背景）

本地原始 `train_logs/brainx/last.pth` 记录到 epoch 299，即共 300 epochs。
原始跨受试者训练主要设置：

| 设置 | 原始 UMBRAE |
|---|---|
| Subjects | S1/S2/S5/S7 |
| Batch size | 128 |
| Optimizer | AdamW |
| Max LR | `3e-4` |
| Scheduler | OneCycleLR |
| Weight decay | 普通参数 `1e-2`，bias/LayerNorm `0` |
| Mixed precision | fp16 |
| Target | frozen CLIP 256 patch tokens |
| Loss | MSE |
| Epochs | 300（当前主 checkpoint） |

本文中的 3-epoch caption 实验只训练新增 adapter，冻结 BrainX 和 Shikra；
它不是把原始 300-epoch UMBRAE 训练缩短为 3 epochs。

---

## 17. 建议的下一步实验

按优先级：

1. **严格原始 UMBRAE 零训练 baseline**
   - best BrainX checkpoint；
   - frozen original `mm_projector`；
   - 原 prompt、解析和 982 test；
   - 多参考 caption evaluation。

2. **S2/S5/S7 strict caption**
   - 只跑 UMBRAE-only、uniform、soft；
   - 判断 S1 的 uniform 优势是否跨 subject。

3. **多参考 caption evaluation**
   - 每个 COCO image 使用全部 reference；
   - 减少单参考 BLEU/CIDEr 偏差。

4. **提高 ROI diversity，而不是继续调温度**
   - subject/ROI-conditioned low-rank correction；
   - 明确限制 deviation；
   - 必须与 uniform coverage baseline 比较。

5. **任务分离**
   - caption 使用 coverage-oriented routing；
   - retrieval 使用 contrastive-oriented global head；
   - 不要求一个 routing 同时在所有任务最优。

---

## 18. 主要结果路径

| 内容 | 路径 |
|---|---|
| 真实 ROI mappings | `../roi_indices/subjXX_neuroroute_v1.json` |
| Stage-1 四受试者 alignment | `stage1_outputs/cross_subject_summary/` |
| Stage-1 retrieval | `stage1_outputs/retrieval_summary/cross_subject_l24/` |
| Caption ID mapping | `stage2_outputs/caption_mapping/coco73k_captions.json` |
| Generic Stage-2 caption | `stage2_outputs/subj01_real_3epochs/` |
| Strict UMBRAE–NeuroRoute | `umbrae_neuroroute_outputs/subj01_real_3epochs/` |
| Structured routing | `umbrae_neuroroute_outputs/subj01_structured/` |
| Structured routing diagnostics | `umbrae_neuroroute_outputs/subj01_structured/routing_analysis/` |

相关阶段报告：

- `docs/STAGE1_ROUTING_RESULTS.md`
- `docs/STAGE1_RETRIEVAL_RESULTS.md`
- `docs/STAGE2_CAPTION_RESULTS_S1.md`
- `docs/UMBRAE_NEUROROUTE_RESULTS_S1.md`
- `docs/STRUCTURED_ROUTING_RESULTS_S1.md`

本文件是截至 2026-06-23 的统一结论入口；早期状态文档中的 blocker 或
“尚未实现”描述可能已经被后续工作解决。
