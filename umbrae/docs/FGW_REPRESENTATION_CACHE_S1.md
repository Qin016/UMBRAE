# subj01 FGW offline representation cache

生成日期：2026-08-20

## 结论

已使用现有 subj01 Stage-1 soft alignment 最优 checkpoint，对 NSD 训练集的
8,559 个 stimulus 一次性生成离线 ROI/CLIP 表征缓存。缓存只用于后续
RSA、FGW 和稳定性分析；没有重新训练 Stage-1、UMBRAE 或 Shikra，没有运行
caption 实验，也没有修改 ROI mapping 或任何已有实验目录。

缓存根目录：

```text
umbrae/fgw_cache/subj01/train_seed42/
```

使用的 checkpoint：

```text
umbrae/stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt
```

checkpoint SHA-256：

```text
2cad72dc809534709b8bfd12a1d9e6512687fdb9d7b408b311ae66f19d837dea
```

## 实现

新增文件：

- `scripts/cache_roi_clip_representations.py`
- `tests/test_cache_roi_clip_representations.py`

缓存脚本具有以下约束：

- 只恢复 Stage-1 的 `ROITokenizer`、`BrainToCLIPProjector` 和冻结的
  `CLIPLayerBank`；
- 不导入或加载 UMBRAE、Shikra、Stage-2 adapter；
- 不创建 optimizer，不调用训练入口；
- brain 分支只接收 fMRI，image 只进入冻结 CLIP 离线分支；
- 默认拒绝覆盖已经存在的输出目录；
- 先写入独立临时目录，所有完整性检查通过后才原子发布；
- 使用 NumPy float32 `.npy` 文件，支持后续 `mmap_mode="r"`，无需把全部
  表征载入内存。

真实缓存命令为：

```bash
/opt/conda/envs/brainx/bin/python \
  scripts/cache_roi_clip_representations.py \
  --subject subj01 \
  --data-tar nsd/webdataset_avg_split/train/train_subj01_*.tar \
  --roi-indices-path ../roi_indices/subj01_neuroroute_v1.json \
  --stage1-checkpoint \
    stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt \
  --clip-model-name-or-path openai/clip-vit-large-patch14 \
  --selected-clip-layers 4 8 12 16 20 24 \
  --clip-layer-target-dim 1024 \
  --output-dir fgw_cache/subj01/train_seed42 \
  --split-name train \
  --random-seed 42 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

## 数据与固定划分

输入为现有的 18 个 subj01 NSD train tar shard。审计未发现为 RSA/FGW
预定义的 `geometry_fit / feature_cost_fit / heldout_eval` 划分，因此没有改变
NSD 原有 train/val/test 边界，而是只在 8,559 个训练 stimulus 内创建新的
固定 stimulus-level 60/20/20 划分。原有 300-sample validation 和 982-sample
test 均未纳入本缓存。

划分协议：

1. 稳定 ID 优先使用 `coco73k:<id>`；缺失时才回退到 sample key。
2. 对排序后的 unique stable IDs 使用 NumPy `default_rng(seed=42)` 固定置换。
3. 使用 largest-remainder 方法得到整数 split 数量。
4. 同一个 stable ID 的所有记录只能进入同一 split。
5. tensor 行顺序固定为 numeric sample key，其次 source tar。

实际数据中 8,559 行对应 8,559 个唯一 `coco73k` ID，无重复 stimulus 行。

| Analysis split | 样本数 | 比例 | ordered ID SHA-256 |
|---|---:|---:|---|
| `geometry_fit` | 5,135 | 60.0% | `297854b3cbf358f97d425f8ec9c768d27481b461b357463479f2c0ec07d7b1b8` |
| `feature_cost_fit` | 1,712 | 20.0% | `f41d463268fd92a9be673231232642de8e701e91a4aaa8e0915a6fc33075b9ad` |
| `heldout_eval` | 1,712 | 20.0% | `6dfb051f10ce6edd2029fa2c38f237269095341ca5cefae453abe45bc4c9b52e` |

三个 split 的 ID 交集均为 0，ID 并集为 8,559。完整 split ID 列表保存在：

```text
fgw_cache/subj01/train_seed42/metadata.json -> split_ids
```

每个 tensor 行对应的 ID、sample key、原始/局部 sample index、trial、repeat
数和源 tar 路径保存在各 split 的 `metadata.json -> samples`，因此无需依赖
文件名猜测行对应关系。

## 表征定义与形状

ROI 顺序严格固定为：

```text
[V1, V2, V3, hV4, FFA, EBA, PPA, OPA]
```

CLIP 层顺序严格固定为：

```text
[4, 8, 12, 16, 20, 24]
```

CLIP pooling 严格复用 Stage-1：

```text
mean_non_cls_patch_tokens
```

即每层先取 `hidden_states[layer][:, 1:, :]` 的 256 个非 CLS patch token，
经过该 checkpoint 中冻结的 identity projection head 后，对 patch 维求平均。
没有改用 CLS、CLIP pooled output 或其他 pooling。

每个 split 中保存三个数组：

| 文件 | `geometry_fit` | `feature_cost_fit` | `heldout_eval` | dtype |
|---|---|---|---|---|
| `brain_roi_features.npy` | `[5135,8,1024]` | `[1712,8,1024]` | `[1712,8,1024]` | float32 |
| `projected_roi_features.npy` | `[5135,8,1024]` | `[1712,8,1024]` | `[1712,8,1024]` | float32 |
| `clip_layer_features.npy` | `[5135,6,1024]` | `[1712,6,1024]` | `[1712,6,1024]` | float32 |

定义：

- `brain_roi_features`：`BrainToCLIPProjector` 之前的原始 ROI token；
- `projected_roi_features`：经过共享 Stage-1 brain-to-CLIP MLP 的 ROI token；
- `clip_layer_features`：冻结 CLIP 六层的 mean-patch 表征。

后续主 FGW 分析应使用 `projected_roi_features` 对
`clip_layer_features`；`brain_roi_features` 用于表示选择敏感性分析。缓存没有
保存已经按 learned router 混合过的 `routed_targets`，避免对 ROI-layer
correspondence 产生循环定义。

## Repeat 与 trial metadata

全部 8,559 个样本均有 `number_of_repeats` 和 scalar `trial_id`。训练集总体
repeat 分布为：

| 有效 repeat 数 | 样本数 |
|---:|---:|
| 1 | 307 |
| 2 | 935 |
| 3 | 7,317 |

各 split：

| Split | repeat=1 | repeat=2 | repeat=3 |
|---|---:|---:|---:|
| `geometry_fit` | 191 | 568 | 4,376 |
| `feature_cost_fit` | 52 | 196 | 1,464 |
| `heldout_eval` | 64 | 171 | 1,477 |

为严格复现 Stage-1 checkpoint 的输入分布，本缓存对
`fmri[:number_of_repeats]` 先转 float32，再求平均，然后进入 ROI tokenizer。
本步骤只缓存 mean-input 主表征，没有把单独 repeat-wise ROI token 写入缓存。
因此 trial/repeat metadata 可用于筛选或后续新增的噪声分析，但当前三个数组
本身不是 repeat-resolved 表征。`trial_id` 仍然只是每个 WebDataset sample
一个 scalar，不能被解释为每个 repeat 的 session/run ID。

## 文件布局

```text
fgw_cache/subj01/train_seed42/
├── cache_config.json
├── metadata.json
├── geometry_fit/
│   ├── brain_roi_features.npy
│   ├── projected_roi_features.npy
│   ├── clip_layer_features.npy
│   ├── metadata.json
│   └── cache_config.json
├── feature_cost_fit/
│   └── ... same five files
└── heldout_eval/
    └── ... same five files
```

根和 split 级 `cache_config.json` 均记录：ROI/layer 顺序、checkpoint 路径、
checkpoint 与 ROI mapping SHA-256、feature dimensions、pooling、样本数、数据
split、tar manifest、repeat reduction、固定划分协议和 leakage flags。

## 磁盘占用

| 路径 | 实际字节数 | 可读大小 |
|---|---:|---:|
| `geometry_fit/` | 465,026,814 | 约 444 MiB |
| `feature_cost_fit/` | 155,052,217 | 约 148 MiB |
| `heldout_eval/` | 155,045,450 | 约 148 MiB |
| 完整缓存 | 775,331,544 | `du -sh`: 740 MiB |

只保存六层 mean-patch features，没有保存 `[N,6,256,1024]` patch bank，避免
约 256 倍的非必要 CLIP 缓存开销。

## 完整性验证

真实缓存发布前及发布后均确认：

- 三类 feature 的 sample 轴一致，总数均为 8,559；
- 三个 split 的 shape 与 `[N,8,1024]` / `[N,6,1024]` 合同完全一致；
- 所有数组均为 float32；
- 所有数组无 NaN 或 Inf；
- 8,559 个 stable stimulus IDs 全部唯一；
- split 两两无 ID 交集，且并集覆盖全部 8,559 IDs；
- 每个 split 保存 tensor 行顺序的 SHA-256；
- checkpoint 和 ROI mapping 均保存路径及 SHA-256；
- 输出目录在运行前不存在，未覆盖任何已有结果。

合成测试覆盖：8 ROI、6 CLIP layers、匹配 sample IDs、三个 split 的保存 shape、
finite values、metadata、repeat/trial availability、行与 ID 对应关系，以及同一
stimulus 不跨 split。执行结果：

```text
Synthetic representation-cache tests passed
```

当前 `brainx` 环境未安装 `pytest`，因此同时通过直接执行
`tests/test_cache_roi_clip_representations.py` 完成全部断言；两个文件也通过
`py_compile`。

## Leakage 状态

缓存配置明确记录：

```text
offline_analysis_only = true
uses_image_features_only_for_offline_analysis = true
image_features_used_by_brain_branch = false
image_features_exported_to_training_or_inference = false
```

图像只在本脚本内用于生成离线冻结 CLIP 特征；ROI 表征只由 fMRI 生成。缓存
没有连接 UMBRAE/Shikra/Stage-2，没有写入训练或推理路径，也没有运行任何旧
实验。后续分析必须保持 `heldout_eval` 不参与几何定义、feature-cost 选择或
FGW 超参数选择。
