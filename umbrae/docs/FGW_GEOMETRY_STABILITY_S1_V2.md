# subj01 表征几何稳定性研究 S1 V2

生成日期：2026-08-20

## 修正后结论

```text
GEOMETRY_STATUS = GO
REPEAT_RELIABILITY_STATUS = NOT_AVAILABLE
```

在严格只使用 `offline_discovery`、以 projector 前 ROI token 为 brain 主表征、
使用无放回独立 stimulus 子集和逐 ROI 独立 stimulus-permutation null 后，subj01
的 ROI 表征关系结构仍具有很高的 sampling stability，并明显区别于 null。因此
原 Prompt-2 的“可以继续离线 FGW”工程结论没有改变。

解释边界发生了重要修正：当前证据是 **stimulus-subset stability**，不是 neural
measurement reliability。现有缓存不支持 repeat-based measurement reliability，
所以 GO 不能扩展为 fMRI 测量噪声已得到控制的神经科学结论。

本文始终把 `C_brain` 称为 **ROI representational geometry relation matrix
（ROI 表征几何关系矩阵）**。它不是解剖距离，本文不作层级推断。

## 输入、实现与数据隔离

新增实现：

```text
scripts/analyze_representation_geometry_v2.py
```

输入 manifest：

```text
fgw_outputs/subj01/split_manifest_v1/fgw_stimulus_manifest.json
```

真实运行只加载 `offline_discovery` 的 5,135 个 row index。因为修复后的成员关系
精确复现原缓存 `geometry_fit`，实际只打开了原 `geometry_fit` feature 文件；
`offline_validation` 和 `offline_test` 的 feature 均未加载。

主 brain representation：

```text
roi_tokens_before_projector
storage file = brain_roi_features.npy
shape = [5135, 8, 1024]
```

它是固定维 ROITokenizer 输出，不是 raw ROI voxel vector。V2 不再把
`projected_roi_features` 作为主几何。CLIP representation 仍严格使用：

```text
clip_layer_features.npy
shape = [5135, 6, 1024]
layers = [4, 8, 12, 16, 20, 24]
pooling = mean_non_cls_patch_tokens
```

从 `offline_discovery` 固定抽取 1,500 个唯一 anchor；ID hash 为：

```text
8a807193d5b26994d838a3fce6ddb62473c632c6da12765fd4a1e2380a8badfd
```

完整 stimulus RDM 只临时驻留内存且不落盘。运行约 301 秒，原 Prompt-2 输出
没有修改。

## Primary geometry

定义保持为：

```text
D_brain_r(i,j) = 1 - cosine(z_i,r, z_j,r)
D_clip_l(i,j)  = 1 - cosine(v_i,l, v_j,l)

C_brain[r,r'] = 1 - Spearman(vec_upper(D_brain_r), vec_upper(D_brain_r'))
C_clip[l,l']  = 1 - Spearman(vec_upper(D_clip_l),  vec_upper(D_clip_l'))
```

`C_brain` shape 为 `[8,8]`，`C_clip` shape 为 `[6,6]`，均为 finite、对称矩阵且
对角线为 0。Brain 非对角元素范围为 `[0.3784, 0.9394]`；CLIP 为
`[0.1340, 0.8522]`。较小值只表示两个节点的 stimulus-RDM 排序更相似。

主矩阵、CSV 和修正热图保存在：

- `brain_geometry.npy` / `brain_geometry.csv`；
- `clip_geometry.npy` / `clip_geometry.csv`；
- `brain_geometry_heatmap.png`；
- `clip_geometry_heatmap.png`。

## Stimulus-subset stability

执行 100 个 replicate。每个 replicate 从 1,500-anchor pool 随机抽取两个互斥
的 600-stimulus 子集；每个子集内部无放回，并分别重新估计完整 relation matrix。
这测量 stimulus sampling stability，不是 repeat-based measurement reliability。

| 分支 | upper-triangle Spearman mean | median | 95% CI | Frobenius cosine mean | normalized Frobenius distance mean |
|---|---:|---:|---:|---:|---:|
| Brain | 0.9896 | 0.9907 | [0.9800, 0.9951] | 0.99966 | 0.01466 |
| CLIP | 0.9928 | 0.9929 | [0.9857, 1.0000] | 0.99934 | 0.02240 |

Brain 的 `1 - normalized Frobenius distance` 均值为 `0.9853`，95% 区间
`[0.9786, 0.9907]`。完整 replicate 指标在
`stimulus_subset_stability.json`，小型 relation matrices 在
`stimulus_subset_geometries.npz`。没有把 bootstrap-with-replacement 作为 V2
主分析。

## 修正后的 geometry null

Primary null 在固定的 500-stimulus anchor 子集上运行 500 次。每次都为 8 个
ROI 分别生成独立 stimulus-identity permutation，再重新计算 ROI-system
relation matrix。它破坏跨 ROI 的共同 stimulus 组织，同时近似保留每个 ROI
自己的 feature marginal。

检验统计量是所有 ROI pair 的平均 RDM Spearman correlation，即
`mean(1-C_brain[r,r'])`：

| 统计量 | 数值 |
|---|---:|
| Real shared organization strength | 0.27415 |
| Null mean | -0.00022 |
| Null 95% CI | [-0.00368, 0.00342] |
| 单侧 permutation p | 0.001996 |

真实值高于全部 500 个 null replicate。另对每个 ROI 各执行 100 次 single-ROI
permutation：真实 ROI-to-other mean RDM correlation 范围为
`[0.2075, 0.3087]`，对应 null mean 全部接近 0，八个单侧 p 均为
`0.009901`。这表明结果不是只有一个 ROI 保留 shared stimulus organization。

完整统计和 seed 信息在 `geometry_null_summary.json`；小型 null relation
matrices 在 `geometry_null_matrices.npz`。

## Repeat-based measurement reliability

```text
REPEAT_RELIABILITY_STATUS = NOT_AVAILABLE
```

现有 feature cache 对每个 stimulus 只保存一行，即有效 fMRI repeat 先求均值再
产生 ROI token。虽然 metadata 有 `number_of_repeats` 和一个 scalar `trial_id`，
但没有逐 repeat ROI token，也没有独立 session/run partition。因此不能从相同
stimulus identities 构造 measurement set A/B。

V2 没有用互斥 stimulus 子集近似 repeat reliability，也没有计算 crossnobis。
状态与原因显式保存于 `repeat_based_measurement_reliability.json`。

## ROI diagnostics 与 confound audit

| ROI | voxels | feature variance | norm mean | subset profile Spearman median | single-ROI real/null mean correlation |
|---|---:|---:|---:|---:|---:|
| V1 | 1350 | 0.02830 | 33.913 | 0.964 | 0.2075 / -0.0005 |
| V2 | 1433 | 0.02790 | 34.101 | 0.893 | 0.2495 / -0.0008 |
| V3 | 1187 | 0.02785 | 33.560 | 0.964 | 0.3087 / 0.0003 |
| hV4 | 687 | 0.02710 | 32.948 | 0.929 | 0.2789 / 0.0012 |
| FFA | 687 | 0.02924 | 32.911 | 1.000 | 0.2760 / 0.0003 |
| EBA | 2238 | 0.03545 | 33.126 | 1.000 | 0.3063 / 0.0014 |
| PPA | 720 | 0.02986 | 33.145 | 0.964 | 0.2614 / 0.0006 |
| OPA | 1532 | 0.03115 | 33.475 | 1.000 | 0.3049 / -0.0005 |

全部 ROI 的 near-zero feature-dimension fraction 为 0，所有 profile stability
中位数为正，没有明显 pathological ROI。

ROI voxel count 与 subset-profile stability 的描述性 Spearman 为 `0.2644`
（双侧 `p=0.5269`），没有明显 voxel-count 主导证据。Feature variance 与
stability 的 Spearman 为 `0.7760`（`p=0.0236`）。后者提示较高 feature
variance 的 ROI 在当前 8-node 诊断中更稳定，应在后续解释中保留为 confound
风险；由于仅有 8 个 ROI 且缺失真正 SNR/repeat reliability，这些量没有被自动
回归掉，也不应被当作确定性结论。

完整值见 `roi_diagnostics.csv` 和
`geometry_config.json -> roi_confound_audit`。

## 原 Prompt-2 与 V2 的差异

| 项目 | 原 Prompt-2 | V2 修正 |
|---|---|---|
| Brain 主表征 | projected ROI tokens | projector 前 `roi_tokens_before_projector` |
| 划分来源 | 原 `geometry_fit` | corrected manifest 的 `offline_discovery` |
| 子集分析名称 | `split_half_reliability` | `stimulus_subset_stability` |
| 子集估计 | 单次 500/500 split；另做有放回 bootstrap-to-full | 100 对互斥 600/600 子集，全部无放回 |
| Neural reliability 含义 | 文档虽有限制，但名称有歧义 | 明确不是 measurement reliability；repeat 状态为 NOT_AVAILABLE |
| Primary null | 对 split-B relation matrix 做 node-label permutation | 每个 ROI 独立 permute stimulus identities 后重估 geometry |
| Single-node null | 无 | 每个 ROI 100 次 single-ROI permutation |
| 保留集控制 | 只读旧 `geometry_fit` | 由 ID manifest 强制只读 `offline_discovery` |

原分析实际使用的是 node-label permutation，不是“所有 ROI 共用同一个 stimulus
permutation”；但它检验的是 ROI 标签对齐，而没有直接破坏 shared stimulus
organization。V2 null 改为修正说明要求的 estimand。

两版矩阵并非完全相同，因为 brain representation 和 anchor 都改变。两版 brain
relation matrix 上三角 Spearman 为 `0.8944`；CLIP 为 `0.9893`。原分析报告的
单次 brain disjoint-subset Spearman 为 `0.9529`，V2 的 100-replicate 中位数为
`0.9907`。因此稳定性与 null 证据仍支持 GO，科学结论方向没有改变；改变的是
统计名称、null 定义、主表征和解释边界。

原 Prompt-2 的 bootstrap-with-replacement 目录继续保留，只能作为 secondary
legacy diagnostic：

```text
fgw_geometry/subj01/projected_cosine_spearman_seed42_n1000_b100/
```

## Revised gate

V2 的 GO 条件及结果：

1. Brain independent subset-pair Spearman 中位数至少 0.5 且 95% 区间下界
   大于 0：**通过**；
2. Real shared organization strength 高于 independently permuted ROI null 的
   97.5% 分位且单侧 `p<=0.05`：**通过**；
3. 至多一个 ROI 的 subset-profile median 不大于 0，且无大面积 near-zero
   dimensions：**通过，实际为 0 个**。

Repeat reliability 缺失不单独触发 NO-GO，但限制神经科学解释。Feature variance
关联是需要保留的 caution，不足以表明一个 pathological ROI 决定整体结果。

## 输出与停止点

修正输出目录：

```text
fgw_outputs/subj01/geometry_v2/
```

包含所有要求的主矩阵、`stimulus_subset_stability.json`、
`repeat_based_measurement_reliability.json`、`geometry_null_summary.json`、
`roi_diagnostics.csv` 和修正热图；另保存可复核的小型 replicate/null relation
matrices。目录 `du -sh` 为 516 KiB。

本步骤没有重新提取 feature，没有读取 offline validation/test feature，没有修改
Stage-2，也没有实现 FGW。Prompt-0R–2R 修正到此停止。
