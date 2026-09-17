# subj01 表征几何可靠性研究（S1）

生成日期：2026-08-20

## 初步结论

**GO：subj01 的 ROI 表征几何具有足够的初步可重复性，可以进入离线
srFGW correspondence 实现与验证。**

此结论只说明当前固定表征、固定 stimulus 集上的关系结构可靠性通过了预注册式
门槛。本文中的 `brain_geometry` 始终称为 **ROI representational geometry
relation matrix（ROI 表征几何关系矩阵）**；它不是解剖距离，也不支持任何皮层
层级结论。

GO 的实际证据：

- brain split-half relation-matrix Spearman：`0.9529`；
- brain 节点标签置换零分布：均值 `-0.0008`，95% 区间
  `[-0.2764, 0.5099]`，单侧 `p=0.000999`，相对零分布 `z=5.05`；
- brain bootstrap-to-full Spearman：均值 `0.9810`，95% 区间
  `[0.9658, 0.9929]`；
- 8 个 ROI 的 split-half profile Spearman 全部为正，范围
  `[0.6429, 0.9643]`，没有看到结果只由一两个不稳定 ROI 主导的证据；
- CLIP sanity check 更稳定：split-half `0.9964`，bootstrap-to-full 95%
  区间 `[0.9893, 1.0000]`，符合冻结图像表征应比 fMRI 稳定的预期。

GO 判据在运行前由脚本固定为：brain split-half 相关必须超过节点标签置换零分布
的 97.5% 分位数且单侧 `p<=0.05`；同时 bootstrap-to-full Spearman 中位数
必须至少为 `0.5`，95% 区间下界必须大于 0。本次两个条件均通过。

## 实现与运行范围

新增实现：

```text
scripts/analyze_representation_geometry.py
```

脚本只加载以下缓存 split：

```text
fgw_cache/subj01/train_seed42/geometry_fit/
```

没有读取 `feature_cost_fit` 或 `heldout_eval`，没有加载 UMBRAE、Shikra 或
CLIP 模型，没有修改 Stage-2，也没有重新训练或运行任何已有 caption 实验。

真实运行命令：

```bash
/opt/conda/envs/brainx/bin/python \
  scripts/analyze_representation_geometry.py \
  --representation-cache fgw_cache/subj01/train_seed42 \
  --cache-split geometry_fit \
  --output-dir \
    fgw_geometry/subj01/projected_cosine_spearman_seed42_n1000_b100 \
  --geometry-type cosine_rsa_spearman \
  --brain-feature projected_roi_features \
  --bootstrap-count 100 \
  --stimulus-subset-size 1000 \
  --null-count 1000 \
  --random-seed 42 \
  --device cuda
```

运行耗时约 119 秒。输出目录拒绝覆盖已存在路径。

## 几何定义

主分析使用 Stage-1 soft checkpoint 产生的 `projected_roi_features`，shape 为
`[5135, 8, 1024]`；CLIP 使用 `clip_layer_features`，shape 为
`[5135, 6, 1024]`。CLIP 层和 pooling 完全继承缓存：

```text
layers  = [4, 8, 12, 16, 20, 24]
pooling = mean_non_cls_patch_tokens
```

ROI 顺序为：

```text
[V1, V2, V3, hV4, FFA, EBA, PPA, OPA]
```

从 5,135 个 `geometry_fit` stimulus 中由 NumPy `SeedSequence(42)` 派生的
独立 RNG 固定抽取 1,000 个唯一 anchor。按缓存行排序后的 anchor stable-ID
SHA-256 为：

```text
107de8f0be26c25f6addda319ef621e5d627e64da9d644ecef811d846c4d8d93
```

对每个 ROI `r`：

```text
D_brain_r(i,j) = 1 - cosine(z_i,r, z_j,r)
```

对每个 CLIP layer `l`：

```text
D_clip_l(i,j) = 1 - cosine(v_i,l, v_j,l)
```

只提取每个 RDM 的严格上三角向量，随后计算：

```text
C_brain[r,r'] = 1 - Spearman(vec(D_brain_r), vec(D_brain_r'))
C_clip[l,l']  = 1 - Spearman(vec(D_clip_l),  vec(D_clip_l'))
```

最终 shape 分别为 `[8,8]` 和 `[6,6]`，对称且对角线为 0。完整 stimulus
RDM 只以 float32 临时驻留内存，从未写盘；最终只保存小型 relation matrix、
bootstrap relation matrices 和固定 anchor ID。

## 主矩阵结果

brain relation matrix 的非对角元素范围为 `[0.5305, 0.9548]`；CLIP relation
matrix 的非对角元素范围为 `[0.1249, 0.8748]`。较小值表示两个节点的
stimulus-RDM 排序更相似。这个量只表示表征关系，不表示 ROI 的物理位置。

完整数值见：

- `brain_geometry.csv` / `brain_geometry.npy`
- `clip_geometry.csv` / `clip_geometry.npy`
- `brain_geometry_heatmap.png`
- `clip_geometry_heatmap.png`

## Split-half 与 null

1,000 个 anchor 使用独立派生 RNG 随机分为两个互斥的 500-stimulus 子集；
两侧分别重新估计完整 relation matrix，然后比较其严格上三角。

| 分支 | split-half Spearman | null mean | null 95% CI | 单侧 p | z vs null |
|---|---:|---:|---:|---:|---:|
| Brain | 0.9529 | -0.0008 | [-0.2764, 0.5099] | 0.000999 | 5.05 |
| CLIP | 0.9964 | -0.0021 | [-0.4143, 0.6179] | 0.004995 | 3.73 |

每次 null replicate 都整体置换 split-B 的节点标签，同时保留矩阵内部结构。
这比独立打乱矩阵元素更贴合“同名 ROI/layer 对齐是否优于随机标签”的问题。
完整 split 矩阵、anchor position、ID hash 和 1,000 个 null 值保存在
`split_half_reliability.json`。

## Bootstrap 稳定性

执行 100 次 stimulus bootstrap；每次从 1,000 个 anchor 中有放回抽取 1,000
次，独立重估 relation matrix。没有把 RDM pair 当作独立样本。

| 分支 | bootstrap-to-full mean | median | 95% CI |
|---|---:|---:|---:|
| Brain | 0.9810 | 0.9819 | [0.9658, 0.9929] |
| CLIP | 0.9959 | 0.9964 | [0.9893, 1.0000] |

Brain 单个非对角 relation entry 的 bootstrap 标准差范围为
`[0.0105, 0.0188]`，均值 `0.0145`；平均 95% 区间宽度为 `0.0542`。
CLIP 对应标准差范围为 `[0.0043, 0.0201]`，均值 `0.0152`；完整结构的
排序稳定性仍高于 brain。

均值、方差和逐元素 95% 区间保存在 `bootstrap_geometry_summary.json`；100
个完整的小型 relation matrix 保存在 `bootstrap_geometries.npz`。后者不是
stimulus RDM。

## ROI diagnostics

| ROI | voxel 数 | feature variance 均值 | norm 均值±SD | split-half profile ρ | bootstrap profile ρ mean |
|---|---:|---:|---:|---:|---:|
| V1 | 1350 | 0.000556 | 5.366±0.120 | 0.893 | 0.953 |
| V2 | 1433 | 0.000540 | 5.387±0.108 | 0.857 | 0.933 |
| V3 | 1187 | 0.000580 | 5.374±0.116 | 0.643 | 0.859 |
| hV4 | 687 | 0.000580 | 5.386±0.108 | 0.929 | 0.913 |
| FFA | 687 | 0.000697 | 5.383±0.139 | 0.964 | 0.990 |
| EBA | 2238 | 0.000862 | 5.407±0.130 | 0.964 | 0.973 |
| PPA | 720 | 0.000776 | 5.568±0.109 | 0.964 | 0.982 |
| OPA | 1532 | 0.000777 | 5.408±0.123 | 0.964 | 0.980 |

voxel 数直接读取缓存记录的真实 subj01 ROI mapping。所有 ROI 的 1,024 个
feature dimensions 均没有 near-zero variance dimension。V3 的 profile
split-half 是最低值，但仍为正且 bootstrap mean 为 0.859；当前没有证据表明
总体结果被单个异常 ROI 驱动。由于每个 ROI profile 只有另外 7 个节点，这一
逐 ROI 相关是粗粒度诊断，不应单独作强推断。详细 min/max、median、区间和
entry uncertainty 见 `roi_reliability_summary.csv`。

## Repeat-aware 几何限制

缓存 metadata 对所有样本均提供 `number_of_repeats` 和 scalar `trial_id`，但
实际 cached ROI tensor 是先对每个 stimulus 的有效 repeats 求均值后得到的一行
表征。缓存没有逐 repeat ROI representation，也没有可用于独立 partition 的
session/run 标签。

因此本研究**没有计算 crossnobis、cross-validated distance 或 repeat-wise
SNR**。仅知道 repeat 数量不足以正确恢复这些统计量，不能从均值表征伪造
noise-aware geometry。若后续需要该分析，必须新增一个独立、明确保存
repeat-resolved ROI token 和 session/run partition 的缓存版本；这不应改写本次
S1 缓存。

## 产物与完整性

输出目录：

```text
fgw_geometry/subj01/projected_cosine_spearman_seed42_n1000_b100/
```

包含：

```text
anchor_ids.json
bootstrap_geometries.npz
bootstrap_geometry_summary.json
brain_geometry.csv
brain_geometry.npy
brain_geometry_heatmap.png
clip_geometry.csv
clip_geometry.npy
clip_geometry_heatmap.png
geometry_config.json
roi_reliability_summary.csv
split_half_reliability.json
```

目录总文件大小 348,943 bytes，`du -sh` 为 376 KiB。已确认：

- brain/CLIP 主矩阵 shape 分别严格为 `[8,8]` / `[6,6]`；
- 矩阵 finite、对称且对角线为 0；
- anchor ID 共 1,000 个且全部唯一；
- ROI/layer 顺序和 CLIP pooling 写入 `geometry_config.json`；
- 不保存完整 N×N RDM；
- `feature_cost_fit` 与 `heldout_eval` 未读取；
- 没有 Stage-2 变更。

实现先通过 64 个合成 stimulus、8 ROI、6 CLIP layer 的 shape、对称性、
split-half null 和 bootstrap 检查；脚本通过 `py_compile`。当前环境未安装
`pytest`，所以验证采用直接合成断言与真实输出完整性检查。

## 下一步边界

该 GO 只授权下一 prompt 中的离线 FGW 实现，不表示 FGW 已经实现或验证。
后续 correspondence 超参数应只使用 `geometry_fit` 和
`feature_cost_fit`，最终结论只在此前未读取的 `heldout_eval` 上评估。本步骤
按要求停止于几何可靠性研究，未实现 FGW，也未修改 UMBRAE/Shikra/Stage-2。
