# FGW correspondence validation — subj01 / S1 / V1

## 结论

`VALIDATION_STATUS = READY_FOR_OFFLINE_TEST`

该状态严格按 Prompt 4A 运行前写入的阈值判定，仅表示 **subj01 的探索性（EXPLORATORY）离线结构对应分析可进入一次性 offline_test**，不构成跨被试科学结论，也不授权 Stage-2。Prompt 4A 没有读取 `offline_test` 特征，没有运行 Stage-2，也没有创建 `best_transport_plan.npy`。

锁定的配置是：

| 字段 | 值 |
|---|---:|
| method | `sr_fgw` |
| structure weight β | 0.50 |
| coverage weight λ | 0.00 |
| entropy coefficient | 0.00 |
| configuration selection | 最低固定 validation composite |
| within-config initialization selection | 仅按 discovery objective |
| best discovery initialization | `random`, seed 1006 |

这里只锁定配置，不锁定或发布最终 transport plan。完整锁定记录见 `fgw_locked_configuration.json`。

## 冻结定义与数据边界

- discovery：5,135 个唯一 stimulus；仅用于 probe、T 和各类拟合 null。
- validation：1,712 个唯一 stimulus；仅用于配置选择和评价，未用于同一配置内部的初始化选择。
- primary validation geometry：从 validation 独立、无放回选取固定 1,500 anchors（seed 43001），使用 geometry_v2 的 `1 - cosine` RDM 和严格上三角 RDM 向量间的 `1 - Spearman`。
- brain 表示：projector 前 ROI token，形状 `[N, 8, 1024]`；不是 raw voxel vector。
- CLIP 表示：`mean_non_cls_patch_tokens`，层顺序 `[4, 8, 12, 16, 20, 24]`，形状 `[N, 6, 1024]`。
- ROI 顺序：`[V1, V2, V3, hV4, FFA, EBA, PPA, OPA]`。
- validation feature probe：对每个 ROI 用全部 discovery 重新拟合 Prompt-3 固定 ridge（α=100）；X 的中心/尺度及 Y 的中心仅由 discovery 拟合，再冻结并预测 validation。
- 固定尺度：`s_feature=0.18190438003847248`，`s_gw=0.202123349231981`，未在 validation 重估。
- 固定评分：`0.5 * feature_normalized + 0.5 * gw_normalized`；coverage penalty 不进入 validation 分数。

validation 矩阵形状为：`M_validation [8,6]`、`C_brain_validation [8,8]`、`C_clip_validation [6,6]`。`C_brain` 始终称为“ROI representational geometry relation matrix”，不是解剖距离，也不据此声称层级。

## 候选配置与收敛

预注册网格共 21 个配置，包括：feature routing、4 个 feature+coverage、balanced outer OT、3 个无 coverage srFGW、以及 3×4 个 srFGW+coverage。每个可微结构配置使用相同 20 个初始化：uniform、feature-informed、random seeds 1001–1018。

`source_constrained_feature_coverage λ=0.03` 的 20 个初始化均未满足冻结收敛准则，因此记为 `NO_CONVERGED_INITIALIZATION`；没有替换 λ、改变优化器或放宽容差。同 coverage 的 λ=0.03 对照在主比较文件中明确记为 `BASELINE_NOT_CONVERGED`。

## 选中配置的 held-out 表现

| 指标 | srFGW β=.5 | 最强非结构基线：feature+coverage λ=1 | balanced outer OT |
|---|---:|---:|---:|
| validation feature normalized | 0.800065 | 0.884005 | 0.940151 |
| validation GW normalized | 0.308794 | 0.567161 | 0.550161 |
| validation composite | 0.554430 | 0.725583 | 0.745156 |
| target entropy | 1.4942 | 1.7620 | 1.7918 |
| effective target layers | 4.456 | 5.824 | 6.000 |
| mean row entropy | ≈0 | 0.2959 | 0.2965 |
| mean pairwise ROI JS | 0.5941 | 0.5955 | 0.6097 |

200 次 validation 75% 无放回子采样中，选中配置的 composite 为 0.55618，95% percentile CI `[0.55033, 0.56294]`，CV=0.00640。

相对最强非结构基线（selected minus baseline）的配对 95% CI：

- feature：`[-0.08529, -0.08272]`；
- GW：`[-0.26201, -0.24573]`；
- composite：`[-0.17296, -0.16464]`。

相对 balanced outer OT 的配对 95% CI：feature `[-0.14179, -0.13844]`、GW `[-0.24740, -0.22891]`、composite `[-0.19280, -0.18419]`。因此结构改善不能仅由 target coverage 解释。

同 coverage 强度的 9 个可用 `srFGW+coverage` 对 `feature+coverage` 比较中，GW 和 composite 差异的 95% CI 均严格低于 0；feature 项则存在权衡，结构配置通常牺牲一部分直接 feature fit。完整逐 β/λ 结果见 `primary_structure_comparisons.json` 和 CSV。

## 可识别性与 ROI 行稳定性

选中配置 20/20 初始化收敛。1% near-optimal discovery 集含 5 个计划，且恰好等于 top-5：

- pairwise Frobenius distance：median 0.25，范围摘要 CI `[0, 0.25]`；
- mean row correlation：median 0.70，CI `[0.70, 1.00]`；
- mean row JS：median 0.1733，CI `[0, 0.1733]`；
- target marginal cosine：全部 1.0，target marginal L1：全部 0；
- preferred-layer agreement：median 0.75，CI `[0.75, 1.00]`。

因此 near-optimal plans 不是完全无关，满足预注册的整体可识别性阈值，但并非每一行都同样稳定。V1、V2、V3、hV4、EBA 的 near-optimal per-ROI row correlation 为 1；FFA 的 median 为 -0.2，PPA/OPA 的分布也包含 -0.2。应把结论理解为“整体对应可复现，但部分高阶 ROI 的精确层身份仍有局部最优歧义”。没有把不稳定计划平均成平滑 consensus。

最佳 discovery 初始化得到的描述性一对一行偏好为：V1→L4、V2→L4、V3→L12、hV4→L16、FFA→L24、EBA→L24、PPA→L20、OPA→L24。该表只描述 transport，不是视觉皮层或 CLIP 深度层级结论。

## Null 检验

所有拟合 null 均在 discovery 上拟合，并对原始、未置换的 validation 矩阵评价。

| Null | 次数 | null validation GW normalized | real-vs-null 单侧 p |
|---|---:|---:|---:|
| ROI geometry identity mismatch | 200 个唯一置换 | mean 0.6584，95% CI `[0.4352,0.8547]` | 0.00498 |
| CLIP geometry identity mismatch | 200 个唯一置换 | mean 0.5363，95% CI `[0.3426,0.7877]` | 0.00995 |
| independent ROI stimulus identity | 200 | mean 0.5064，95% CI `[0.4974,0.5885]` | 0.00498 |
| matched-marginal row-permutation | 500 | mean 0.6429，95% CI `[0.4361,0.7534]` | 0.00200 |

matched-marginal null 精确保留 source marginal、target marginal、总质量和 row-entropy multiset。它也精确保留 mean pairwise JS=0.5941 和 preferred-layer diversity=5，证明这两个指标单独不能支持 ROI specialization；真正提供证据的是原始 ROI 行身份下更低的 held-out GW，以及 cross-initialization 行稳定性。

feature-pairing null 共 20 次，均用相同 5-fold cross-fitting 重新估计 `M_discovery_null`。其 validation feature normalized mean=0.7093，而真实选中计划为 0.8001（低为好），单侧 `p_real_feature_lower=1.0`。所以本分析**没有证据表明直接 brain-image feature compatibility 在 geometry 之外提供额外 held-out feature 贡献**。选中结果主要由结构项支持；不能把它叙述成 feature probe 和结构同时获得独立验证。

## 注册的方差敏感性

选择完成后只运行一次 `variance_standardized_geometry_sensitivity`。每个 ROI 的 per-dimension mean/std 仅用 discovery 拟合，并原样应用于 validation；该分析未参与 β/λ 选择。

标准化几何下的最佳 discovery 计划与 primary T 完全相同：Frobenius distance=0、cosine≈1、8 行 correlation=1、JS=0、preferred-layer agreement=1。因此主对应对已注册的 feature-variance 处理不敏感。该结果不消除 Prompt-2R 的方差相关混杂，只说明这一项具体敏感性操作没有改变当前 T。

## Prompt 4A 九个问题的明确回答

1. **结构项是否优于 feature+coverage 的 held-out structure？** 是。选中 srFGW 对最强非结构基线的 GW 差为 -0.25434，配对 95% CI `[-0.26201,-0.24573]`；同 coverage 的可用比较也一致。
2. **near-optimal local minima 下 T 是否可识别？** 整体可识别但不完美。median preferred agreement=0.75、median mean-row correlation=0.70；FFA/PPA/OPA 有局部歧义。
3. **真实几何是否优于 shuffled geometry？** 是。三类要求的几何 null 单侧 p 分别为 0.00498、0.00995、0.00498。
4. **直接 feature compatibility 是否在 geometry 外有贡献？** 未观察到；feature-pairing null 的结果反而更低，`p=1.0`。
5. **ROI-specific rows 是否可复现？** 整体达到预注册门槛，5/8 ROI 完全稳定；3 个高阶 ROI 仍不稳定，必须保留该限定。
6. **多层使用是否只是 coverage artifact？** 不是。获选配置 λ=0，仍使用 5 个层；相对 feature+coverage 和 balanced 基线的 held-out GW/composite 优势显著。但 JS/diversity 本身不构成证据。
7. **对 feature-variance sensitivity 是否稳健？** 是；secondary T 与 primary T 完全一致。
8. **offline_test 前锁定哪个配置？** `sr_fgw`, β=0.5, λ=0, entropy=0，solver 与 20 初始化种子完整写入锁定 JSON。
9. **VALIDATION_STATUS？** `READY_FOR_OFFLINE_TEST`，限定为 `EXPLORATORY subj01`。

## 产物与溯源

实现文件：

- `scripts/validate_fgw_correspondence.py`
- `models/fgw_correspondence.py`（增加数值安全的 natural-log JS divergence）
- `tests/test_validate_fgw_correspondence.py`

输出目录：`fgw_outputs/subj01/correspondence_validation_v1/`，总计约 3.1 MiB。必需的 validation matrices、per-config/per-init 计划与指标、local-minimum identifiability、200 次子采样、四类拟合 null、matched-marginal null、方差敏感性、heatmaps、锁定配置和完整 provenance 均已保存。

关键 SHA-256：

- preregistration：`ab6bbb2f61906dcfc77fabc28636bb3baca1ed0781393a076dc5c285310d8902`；运行前后未改变；
- discovery M：`11e217440ed1f47e6ca1f9dbeec8a60c7d4147285c4abb77ff0343f7fba8e1a7`；
- discovery Cbrain：`0ce5e158f71f50fe0ccd619890aee75b2d888f372d619fc9a3dc6bf96842f976`；
- discovery Cclip：`d6524f1537608d1fe6e2e4a719a24432ed22a18757c6d05dea6b8f90abc564bb`；
- validation M：`bd8702ea83ebcf35ba1907473f3c01071a95364dd520a023a0b8a997af97ff37`；
- validation Cbrain：`79583179372f1fc59a2c7a90607a5a3229480b85f98f38f903b14628f65bf1e4`；
- validation Cclip：`27bae98fad30d0ee012f3279b2d6963cd0fe6a67c9974ab98cdeb679daff5f71`。

验证脚本耗时约 1266.9 秒。测试覆盖 simplex/FGW 既有数值测试、固定 validation score、200 个唯一 ROI permutations 和 identifiability 汇总；POT 未安装，因此既有 POT reference 检查按设计跳过。运行时不需要加载 torchvision、CLIP、UMBRAE 或 Shikra。

## 停止边界

Prompt 4A 到此结束。`offline_test` 仍未打开；没有运行 Stage-2，没有修改 UMBRAE/Shikra，没有重跑既有 caption 实验，也没有生成最终 transport plan。
