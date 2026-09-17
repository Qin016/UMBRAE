# FGW locked refit and one-shot offline test — subj01 / S1 / V1

## 最终状态

`OFFLINE_TEST_STATUS = EXPLORATORY_PASS`

该状态严格限于 **subj01 / EXPLORATORY**。它不是跨被试结论，不授权任何解剖层级解释，也没有运行 Stage-2 或 caption evaluation。

锁定配置保持不变：`sr_fgw`、β=0.5、λ_cov=0、entropy=0、`random seed 1006`。最终 fitting set 是 5,135 个 discovery stimulus 与 1,712 个 validation stimulus 的并集，共 6,847 个唯一 stimulus，与 1,712 个 offline_test stimulus 和两套 downstream protected IDs 均无重叠。

## 测试前冻结证明

最终计划只用 6,847 个 final-train stimulus 拟合一次，并在打开 offline_test 索引和特征前保存、哈希和持久化 marker：

`FINAL_PLAN_HASH = 51e65527f3bbc1a3951ec376dd8be3662cab9e6b5f682d2a0da26fb347e20760`

marker UTC 时间：`2026-08-20T14:27:09.897652+00:00`。

文件时间顺序、marker 内容和测试后的重新哈希共同确认：

`final_transport_plan_pretest.npy → final_plan_frozen.marker → offline_test_feature_cost.npy`

最终计划测试后 SHA-256 与 marker 完全一致。baseline 及所有 brain/CLIP identity、independent-ROI 和 feature-pairing null plans 也均在 marker 前拟合并哈希。offline_test 没有参与 probe、geometry、T、超参数或初始化选择。

## 最终 refit

- `M_final_train [8,6]`：在合并的 6,847 个 stimulus 上沿用固定 5-fold、stimulus-ID-level OOF ridge 协议；α=100，fold seed=42。
- offline_test probe：在全部 6,847 个 final-train stimulus 上拟合一次同一 ridge；所有中心/尺度均来自 final-train。
- `C_brain_final_train [8,8]`、`C_clip_final_train [6,6]`：只使用 final-train，沿用 geometry_v2 的 1−cosine RDM、严格上三角 RDM 向量间的 1−Spearman，以及固定 1,500 anchors。
- loss scales：原样复用 Prompt-3 的 `s_feature=0.18190438003847248`、`s_gw=0.202123349231981`；未重新标定。
- 最终 srFGW：只运行锁定的 random seed 1006；98 iterations 后收敛，source-marginal error=0，无负质量、NaN 或 Inf。
- 预选 baseline：Prompt-4A 已指定的 `source_constrained_feature_coverage λ=1`，使用其预选 `random seed 1004`，没有在 offline_test 上重新选择。

## 一次性 offline_test 指标

| 指标 | frozen srFGW | pre-selected baseline | FGW − baseline |
|---|---:|---:|---:|
| raw feature loss | 0.141325 | 0.156521 | -0.015196 |
| normalized feature loss | 0.776916 | 0.860456 | -0.083539 |
| raw GW loss | 0.065635 | 0.116066 | -0.050432 |
| normalized GW loss | 0.324726 | 0.574235 | -0.249509 |
| fixed composite | 0.550821 | 0.717346 | -0.166524 |

所有 loss 均为越低越好。冻结 srFGW 在 feature、structure 和固定 composite 三项均优于预选 baseline。

## Test subsampling uncertainty

对 1,712 个 offline_test stimulus 做了 200 次固定 seed、80%（每次 1,370 个）无放回子采样；每次只重算 evaluation M/C，两个计划始终冻结。

FGW − baseline 的 95% percentile CI：

- normalized feature：`[-0.08470, -0.08208]`；
- normalized GW：`[-0.25915, -0.24434]`；
- composite：`[-0.17137, -0.16383]`。

结构优势和 composite 优势均稳定，且 CI 完全低于零。

## 锁定 null evaluation

所有 null plans 均只用 6,847 个 final-train stimulus 拟合，并对原始、未置换的 offline_test geometry 评价。

| Null | N | null normalized GW mean | 95% interval | real lower 的经验 p |
|---|---:|---:|---:|---:|
| brain-geometry identity mismatch | 200 unique | 0.66333 | `[0.45906,0.82949]` | 0.00498 |
| CLIP-geometry identity mismatch | 200 unique | 0.54592 | `[0.38299,0.79101]` | 0.00498 |
| independent-ROI stimulus permutation | 200 | 0.51351 | `[0.50785,0.53129]` | 0.00498 |

真实冻结计划的 normalized GW=0.32473，显著低于全部三类预注册结构 null 分布，满足 p<0.05 的冻结判据。

## Feature-pairing confirmatory check

20 个 feature-pairing null 的 offline-test normalized feature loss 为 0.778431，真实计划为 0.776916，差异仅约 -0.00151。经验 `p_real_feature_lower=1/21=0.04762`，恰为 20 个 null draw 下的最小可达 p；同时 Prompt-4A validation 结果为 p=1.0。

因此 offline_test 出现了边界性的名义分离，但它与 validation 不一致、效应很小且 null 分辨率有限。整体上仍**不能声称 direct ROI-to-layer feature compatibility 获得了独立验证**。适当解释是：当前对应主要由 representational geometry 支持，而不是由直接 feature compatibility 的独立证据支持。该 null 不改变锁定配置，也不进入 PASS 判据。

## Transport diagnostics

最终计划的 target marginal：

- L4=0.25，L8=0，L12=0.125，L16=0.125，L20=0.125，L24=0.375；
- target entropy=1.49418；
- effective target layer count=4.45566；
- mean row entropy≈0；
- mean pairwise ROI JS=0.59413。

描述性 preferred layers：V1→L4、V2→L4、V3→L12、hV4→L16、FFA→L24、EBA→L24、PPA→L20、OPA→L24。这些只描述 transport，不代表生物学或网络深度层级。

Prompt-4A 的 near-optimal 分析表明 V1/V2/V3/hV4/EBA 行稳定，而 FFA/PPA/OPA 存在局部最优歧义。Prompt-4B 按规定只拟合一次最终计划，不能借 offline_test 运行额外初始化来重新评估或消除该不确定性；所以这些行仍应视为不确定。

## 九个问题的明确回答

1. **计划是否在测试访问前冻结？** 是；文件、marker、mtime 和测试后 hash 均验证通过。
2. **FINAL_PLAN_HASH 是什么？** `51e65527f3bbc1a3951ec376dd8be3662cab9e6b5f682d2a0da26fb347e20760`。
3. **srFGW 是否优于预选非结构 baseline？** 是；composite 差=-0.16652，normalized GW 差=-0.24951。
4. **结构优势在 test subsampling 下是否稳定？** 是；GW 差异 95% CI=`[-0.25915,-0.24434]`。
5. **真实 geometry 是否优于预注册 test nulls？** 是；三类结构 null 均 p=0.00498。
6. **feature-pairing null 是否仍不受支持？** 作为稳健、独立证据仍不受支持；test 只有最小可达 p=0.04762 的边界结果，且与 validation p=1.0 冲突。
7. **哪些 ROI 行仍不确定？** FFA、PPA、OPA；这是 4A near-optimal 分析留下的初始化歧义。
8. **OFFLINE_TEST_STATUS？** `EXPLORATORY_PASS`。
9. **是否仍严格限于 subj01 / exploratory？** 是，不能外推到其他被试。

## 输出与停止边界

实现文件：`scripts/run_fgw_offline_test.py`；边界测试：`tests/test_run_fgw_offline_test.py`。

结果目录：`fgw_outputs/subj01/correspondence_offline_test_v1/`。14 个要求的核心文件全部存在，另保存了 final-train OOF matrices、baseline plan、所有 pretest null plans/permutations 和 anchor IDs，便于审计。总大小约 2.2 MiB，完整运行耗时约 899.4 秒。

完整性检查确认：18 个 NPY 文件均无 NaN/Inf；主计划形状 `[8,6]`、每行质量 1/8、总质量 1；brain/CLIP identity permutations 各 200 个且无重复；子采样恰为 200 次；仓库内没有 `best_transport_plan.npy`。

Prompt 4B 到此停止。没有运行 Stage-2，没有创建 caption 结果，也没有根据 offline_test 修改任何计划或配置。
