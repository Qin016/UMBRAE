# FGW stimulus split 与 leakage provenance 修复（S1）

生成日期：2026-08-20

## 结论

Prompt-0R 硬门槛结果为 `AUDIT_STATUS = VERIFIED_READY`，因此执行了本修复。
已在不重新运行 CLIP、ROITokenizer 或 BrainToCLIPProjector 的情况下，建立
subj01 唯一 stimulus 级 manifest 和三个离线 index split。

输出目录：

```text
fgw_outputs/subj01/split_manifest_v1/
```

实现与测试：

- `scripts/build_fgw_stimulus_manifest.py`
- `tests/test_build_fgw_stimulus_manifest.py`

没有删除、移动或重写三个已有 feature 数组。新 `.npy` 文件只是指向原缓存行的
int64 index，不包含复制后的 feature。

## 现有缓存逐行审计

缓存根目录：

```text
fgw_cache/subj01/train_seed42/
```

三个原缓存组共 8,559 行。逐行字段可用性：

| 字段 | 全部行可用 | 实际语义 |
|---|---|---|
| stable stimulus ID | 是 | `coco73k:<id>` |
| nsdId / coco73k ID | 是 | 当前数据实际提供 `coco73k_id`；没有另一个独立 `nsdId` 字段 |
| sample index | 是 | 原始 train cache 全局 `sample_index`，严格覆盖 `0..8558` |
| trial identifier | 是 | 每个 WebDataset sample 一个 scalar `trial_id` |
| subject | 是 | 全部为 `subj01` |
| original source split | 是 | 全部为 NSD `train` |
| repeat information | 是 | `number_of_repeats`，取值 1/2/3 |

当前缓存共有 8,559 个唯一 stimulus，因此重复 stimulus 行数为 0。这里的
repeat 是一个 sample 内的 fMRI measurement row，Prompt-1 在提取 ROI token 前
已对有效 repeat 求均值；它不是缓存中的多行 presentation。manifest 的分组逻辑
仍然按 unique ID 实现，因此若将来缓存出现同一 stimulus 的多行，它们也只能
进入同一个 offline split。

## 下游集合与保护策略

脚本直接扫描实际 subj01 tar 中的 `coco73k.npy`、`trial.npy` 和
`num_uniques.npy`，得到：

| 下游来源 | 行数 | 唯一 ID | 跨来源 ID overlap | 状态 |
|---|---:|---:|---:|---|
| train | 8,559 | 8,559 | 0 | 已被既有 Stage-2 用于训练 |
| validation | 300 | 300 | 0 | 已被反复用于 checkpoint/指标评估，不是 final test |
| test | 982 | 982 | 0 | 受保护的 final caption test 候选，尚未运行 |

已检查 `stage2_outputs/**/adapter_config.json` 和
`umbrae_neuroroute_outputs/**/adapter_config.json`：完成的实验引用 train tar 和
300-sample val tar，没有配置引用 test tar。因此 982-sample NSD test 在当前
仓库记录中是 `PROTECTED_CANDIDATE_NOT_YET_EVALUATED`。这并不把 300-sample
validation 冒充 final test。

所有 offline split 都排除了 validation 和受保护 test ID。实际本地 train、val、
test 的 ID 本来就是两两不交，因此没有丢弃缓存训练行。受保护 test ID 对
`offline_discovery`、`offline_validation` 和 `offline_test` 的 overlap 全部为 0；
它们不会参与后续 `M`、`C_brain`、`C_clip`、FGW 超参数选择或最终 transport-plan
估计。

需要区分两种 held-out：`offline_test` 只对 **FGW 方法选择** 保持未使用；它的
stimulus 属于 Stage-1/Stage-2 train 来源，因而不是 representation-learning 或
caption-training 意义上的 untouched test。真正受保护的 caption test 是上述
982 个未缓存 ID。

## 离线唯一 stimulus 划分

协议：对 FGW-eligible stable ID 做 lexical sort，使用 NumPy
`default_rng(seed=42)` 固定置换，再以 largest-remainder 得到 60/20/20 数量。
划分单位只能是 unique stable stimulus ID，不允许 trial-row 级划分。

| Split | 唯一 stimulus | 缓存行 | sorted-ID SHA-256 | 用途 |
|---|---:|---:|---|---|
| `offline_discovery` | 5,135 | 5,135 | `1da5b47facee500b3749f1a973e98a0ffb38fb357dd00e64aa7944eae0d024c6` | 几何/候选 coupling discovery |
| `offline_validation` | 1,712 | 1,712 | `a9285c0b10fe9d627d19386b47ff948d0343c8c5a32b305bebe941229ff0ecd5` | FGW 超参数选择 |
| `offline_test` | 1,712 | 1,712 | `9ea8456faf5338b1f42a8d9e49924fd71e885999f1b5af5e285e9a0c2e4e4f5e` | 最终一次性离线科学评估 |

零 overlap 检查：

- discovery / validation：0；
- discovery / offline test：0；
- validation / offline test：0；
- 任一 offline split / 300-sample downstream validation：0；
- 任一 offline split / 982-sample protected downstream test：0；
- repeated-stimulus group 跨 split：0；
- eligible ID 覆盖次数：严格一次。

新成员关系恰好分别复现原 `geometry_fit`、`feature_cost_fit`、`heldout_eval`
成员关系。这不是假定：已逐 ID 比较三个集合。区别在于新 manifest 明确给出
用途、保护集合、global-to-local cache row 地址和 leakage 语义。

## 文件与 index 语义

```text
fgw_outputs/subj01/split_manifest_v1/
├── fgw_stimulus_manifest.json
├── cache_feature_aliases.json
├── offline_discovery_indices.npy
├── offline_validation_indices.npy
└── offline_test_indices.npy
```

三个 index 数组的值是 `fgw_stimulus_manifest.json -> cache_row_table` 中的
`global_cache_row`。每个 table entry 再给出：

- 原缓存组 `cache_split`；
- 原数组局部行 `cache_split_row`；
- stable/coco73k ID；
- sample key、trial ID、repeat 数；
- subject、原始 train split 和 source tar。

因此后续脚本可以 mmap 原有三个 feature 数组并按地址 gather，无需复制 740 MiB
缓存。验证已逐行确认 table 中的 stable ID 与原 split metadata 完全一致，并确认
所有三个 feature 文件的局部 index 均在有效范围内。

## ROI token 术语修正

原文件名保持不变，以免破坏现有代码：

```text
brain_roi_features.npy
```

其规范语义别名现在是：

```text
roi_tokens_before_projector
```

这是 `[N,8,1024]` 的固定维 `ROITokenizer` 输出，不是每个 ROI 的 raw voxel
vector。`projected_roi_features.npy` 则是 BrainToCLIPProjector 之后的 token。
别名写入：

- `fgw_outputs/subj01/split_manifest_v1/cache_feature_aliases.json`；
- `fgw_stimulus_manifest.json -> feature_semantic_aliases`；
- 原缓存根新增的非破坏性 sidecar
  `fgw_cache/subj01/train_seed42/cache_feature_aliases.json`。

## 测试与停止点

合成测试覆盖：offline split 唯一 ID 零重叠、同 stimulus 多行不跨 split、受保护
test ID 零 overlap，以及 global/local index 精确重建原数组行。真实 manifest
另通过了 8,559 行全量 metadata/index 核验。脚本和测试通过 `py_compile`，直接
测试结果为：

```text
Synthetic FGW stimulus-manifest tests passed
```

本修复没有运行特征提取、训练或 caption 实验。Prompt-1R 到此结束；Prompt-2R
只允许读取 `offline_discovery`，不得读取 `offline_validation` 或 `offline_test`。
