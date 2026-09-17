# subj01 Offline srFGW Correspondence Discovery（S1）

生成日期：2026-08-20

## 范围与数据硬门槛

本报告只回答 Prompt-3 discovery-stage 问题。全部 fitting、OOF probe、loss-scale
calibration、候选计划估计和初始化选择只使用 `offline_discovery` 的 5,135 个
唯一 stimulus。

以下数据未加载：

- `offline_validation`；
- `offline_test`；
- 既有 300-sample validation；
- 受保护 982-sample test。

没有更新 ROITokenizer、BrainToCLIPProjector、CLIP、UMBRAE 或 Shikra，没有运行
caption training，没有修改 Stage-2。没有生成 `best_transport_plan.npy`，也没有
跨候选宣布科学意义上的最佳方法。

主要实现：

- `scripts/build_fgw_feature_cost_v2.py`
- `models/fgw_correspondence.py`
- `scripts/run_fgw_correspondence_discovery.py`
- `tests/test_fgw_correspondence.py`

输出：

```text
fgw_outputs/subj01/feature_cost_v2/
fgw_outputs/subj01/correspondence_discovery_v1/
```

原 Prompt-2 legacy geometry 未被使用。冻结 geometry 只从修正后的
`fgw_outputs/subj01/geometry_v2/` 加载。

## 1. Cross-fitted M 是否成功估计？

是。Brain input 为 `roi_tokens_before_projector [5135,8,1024]`，CLIP targets 为
严格缓存的 `[5135,6,1024]` mean-non-CLS-patch layer representations，层顺序：

```text
[L4, L8, L12, L16, L20, L24]
```

使用 5-fold unique-stimulus cross-fitting，每 fold 1,027 个 held-out stimulus。
每个 ROI fold 在另外 4,108 个 stimulus 上拟合一个 multi-output linear ridge：

```text
objective = sum_squared_error + 100 * squared_weight_norm
```

`alpha=100` 是全局预设值，没有使用 offline validation，也没有为 48 个 pair
分别调参。每个训练 fold 独立拟合：

- brain dimension mean/std，随后应用于 held-out fold；
- CLIP target mean，target 不做 variance scaling。

Held-out fold 的任何统计量都没有进入 probe fitting。5 个 fold ID hash、source
cache/file hash 和逐 fold 指标保存在 `feature_cost_config.json` 和
`feature_cost_fold_metrics.csv`。

最终 OOF cost：

| ROI | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 0.1699 | 0.2531 | 0.2336 | 0.1852 | 0.2133 | 0.1075 |
| V2 | 0.1653 | 0.2456 | 0.2282 | 0.1822 | 0.2109 | 0.1065 |
| V3 | 0.1622 | 0.2349 | 0.2132 | 0.1717 | 0.2029 | 0.1027 |
| hV4 | 0.1615 | 0.2349 | 0.2132 | 0.1710 | 0.2013 | 0.1018 |
| FFA | 0.1708 | 0.2422 | 0.2112 | 0.1647 | 0.1939 | 0.0980 |
| EBA | 0.1771 | 0.2444 | 0.2069 | 0.1593 | 0.1890 | 0.0956 |
| PPA | 0.1690 | 0.2372 | 0.2050 | 0.1600 | 0.1909 | 0.0965 |
| OPA | 0.1665 | 0.2331 | 0.2014 | 0.1590 | 0.1906 | 0.0967 |

定义严格为：

```text
M[r,l] = mean_i(1 - cosine(predicted_clip_oof[i,r,l], true_clip[i,l]))
```

M shape 为 `[8,6]`，finite，SHA-256 为：

```text
11e217440ed1f47e6ca1f9dbeec8a60c7d4147285c4abb77ff0343f7fba8e1a7
```

已有 soft-routing score 没有用于 M。这些 ridge probes 只是 compatibility
estimators，不是 Stage-1、decoder 或 FGW trainable component。

## 2. M 是否表现出高层偏好？

M 对 L24 有明显的描述性 preference：

| Layer | 跨 ROI mean OOF cost |
|---|---:|
| L4 | 0.1678 |
| L8 | 0.2407 |
| L12 | 0.2141 |
| L16 | 0.1691 |
| L20 | 0.1991 |
| L24 | **0.1007** |

所有 8 个 ROI 的最低 OOF cost 都是 L24。L24 与次低 L4 的 margin 为
`0.0671`，约为 layer-mean cost 标准差的 1.52 倍。因此
`source_constrained_feature_routing` 的每行独立最优解全部选择 L24。

这只表示当前 cross-fitted prediction compatibility；不解释为 model-depth 或
生物学层级。

## 冻结 geometry 与 loss calibration

使用且仅使用 geometry_v2：

```text
C_brain SHA-256 = 0ce5e158f71f50fe0ccd619890aee75b2d888f372d619fc9a3dc6bf96842f976
C_clip  SHA-256 = d6524f1537608d1fe6e2e4a719a24432ed22a18757c6d05dea6b8f90abc564bb
```

已验证 shape `[8,8]` / `[6,6]`、finite、对称、零对角和固定 node 顺序。候选
拟合期间没有用替代定义重算矩阵。

统一参考计划 `T_ref[r,l]=1/48` 给出：

```text
eps_scale = 1e-12
s_feature = 0.18190438003847248
s_gw      = 0.20212334923198100
```

输入尺度：

| Matrix | min | max | mean | std |
|---|---:|---:|---:|---:|
| M | 0.0956 | 0.2531 | 0.1819 | 0.0449 |
| C_brain | 0.0000 | 0.9394 | 0.6360 | 0.2799 |
| C_clip | 0.0000 | 0.8522 | 0.4266 | 0.2827 |

这些 canonical discovery scales 只计算一次，保存在
`loss_scale_calibration.json`；以后 validation/null/subsample 必须复用，不能
逐 run 重新归一化。

## Objective 与 solver

所有 semi-relaxed plans 满足：

```text
T >= 0
sum_l T[r,l] = 1/8
```

Target marginal `m=T^T 1` 自由。Soft coverage 为：

```text
KL(m || Uniform(6)) = sum_l m_l log((m_l + 1e-12)/(1/6))
```

它不保证每层获得非零质量。完整目标：

```text
(1-beta) * L_feature/(s_feature+eps_scale)
+ beta * L_gw/(s_gw+eps_scale)
+ lambda_cov * KL(m||u)
+ epsilon_entropy * sum(T log(T+eps))
```

本次 `epsilon_entropy=0`。Plan entropy 没有作为科学 anti-collapse 参数，也没有
使用 softmax temperature。

自定义 solver 直接做 projected gradient、backtracking 和逐行 Euclidean simplex
projection；允许 exact zeros。每个非 balanced config 使用 uniform、3 个 random
seed 和 feature-informed 共 5 个 initialization，同一 config 内只按最低
offline-discovery objective 选择保存到 `transport_plan.npy` 的初始化。所有
initialization plans、objectives 和 traces 均被保留。

`balanced_outer_ot` 使用 SciPy HiGHS linear programming 精确求解硬 source/target
uniform marginal，不标为 MOT。

## 候选运行摘要

这里只展示 discovery behavior，不作跨候选选择。

| Candidate | beta | lambda_cov | total obj | max target mass | effective layers | init objective range |
|---|---:|---:|---:|---:|---:|---:|
| feature routing | 0 | 0 | 0.5534 | 1.000 | 1.000 | 0.0000 |
| feature coverage | 0 | 0.01 | 0.5713 | 1.000 | 1.000 | 0.0000 |
| feature coverage | 0 | 0.1 | 0.7247 | 0.925 | 1.417 | ~0.0000 |
| balanced outer OT | 0 | hard uniform | 0.9711 | 0.167 | 6.000 | n/a |
| sr-GW | 1 | 0 | 0.2741 | 0.250 | 4.757 | 0.0436 |
| sr-FGW | 0.25 | 0 | 0.6572 | 0.500 | 2.828 | 0.0252 |
| sr-FGW coverage | 0.25 | 0.01 | 0.6647 | 0.500 | 2.828 | 0.0265 |
| sr-FGW coverage | 0.25 | 0.1 | 0.7197 | 0.451 | 4.117 | 0.0238 |
| sr-FGW | 0.5 | 0 | 0.5654 | 0.375 | 4.456 | 0.0424 |
| sr-FGW coverage | 0.5 | 0.01 | 0.5714 | 0.375 | 4.456 | 0.0394 |
| sr-FGW coverage | 0.5 | 0.1 | 0.5918 | 0.375 | 4.782 | 0.0430 |
| sr-FGW | 0.75 | 0 | 0.4375 | 0.250 | 4.757 | 0.0339 |
| sr-FGW coverage | 0.75 | 0.01 | 0.4412 | 0.250 | 4.802 | 0.0356 |
| sr-FGW coverage | 0.75 | 0.1 | 0.4502 | 0.250 | 5.657 | 0.0319 |

由于 beta 不同，表中 total objective 的权重也不同，不能直接按最小值跨方法排序。

## 3. 加入 GW 是否定性改变 coupling？

是。Feature-only 每行独立优化，全部质量流向 L24；它不是 globally
structure-coupled OT。加入 GW 后，候选计划使用多个 layer，并出现不同 ROI row
distribution。例如 beta=0.25 的无 coverage sr-FGW 的描述性 row argmax 为：

```text
[L4, L4, L16, L16, L24, L24, L24, L24]
```

这只是当前 discovery coupling 的描述，不能称作稳定 ROI specialization，也不能
解释为生物学层级。

## 4. Coverage 只改变 target marginal，还是也改变 ROI rows？

两者都可能改变，而且与非凸初始化 basin 交织：

- feature-only 的 `lambda=0.01` 太弱，plan 完全不变；
- feature-only 的 `lambda=0.1` 把 L24 target mass 从 1.0 降到 0.925，并改变
  4 个 ROI 的 row distribution；
- beta=0.25、lambda=0.1 相对无 coverage 的 plan Frobenius distance 为
  `0.1439`，target-marginal distance 也为 `0.1439`；
- beta=0.5、lambda=0.01 的 target marginal 几乎不变，但 plan Frobenius
  distance 为 `0.25`，显示相同/近似 target utilization 下也可能落入不同 ROI-row
  assignment basin。

因此 coverage 不是只对 target marginal 的事后平滑。它是 joint objective 的一
部分，但也不保证 row softness 或非零 target mass。

## 5. balanced_outer_ot 是否与 sr-FGW 明显不同？

是。Balanced outer OT 强制每个 target layer 为 1/6，effective layer count 为
6；它的计划与 beta=0.25 或 beta=0.5 sr-FGW 的 Frobenius distance 都约为
`0.4410`。后两者 target 自由，最大 target mass 分别为 0.50 和 0.375。

这说明硬 global utilization constraint 和 semi-relaxed feature/structure objective
在 discovery data 上产生明显不同的 coupling。不能据此宣布哪一个科学上更好；
那需要后续预先规定的 validation protocol。

## 6. 是否发生 solver 数值 collapse？

没有 NaN、Inf、负质量或 marginal 破坏：

- 所有候选总质量为 1；
- 每个 ROI row 在 tolerance 内严格为 1/8；
- balanced target 在 tolerance 内严格为 1/6；
- exact zeros 被正常保留，coverage 对 zero entries 不产生 NaN。

Feature-only 全部集中 L24 是由 M 的一致性 preference 和 free target marginal
导致的 **objective collapse**，不是数值故障。`lambda=0.01` 不足以改变它。

一个诊断配置需要保留警告：`beta=0.25, lambda=0.01` 的获胜 random-seed-37
初始化在 2,500 次达到 max-iteration，未触发 objective-patience convergence；
其余获胜初始化全部收敛。该配置 finite 且 feasible，但不能称为完全收敛解。

## 7. 初始化是否敏感？

Feature-only families 基本不敏感：所有初始化收敛到同一或数值近似相同 plan。
GW/sr-FGW families 明显敏感：

- initialization objective range 为约 `0.0238–0.0436`；
- 部分 config 的 initialization plan pairwise Frobenius distance 最大达到 `0.5`；
- sr-GW 和多数 sr-FGW 的保存计划来自 random seed 37，而不是 uniform；
- beta=0.5、lambda=0.01 的保存计划来自 uniform，说明没有一个初始化在所有
  config 中占优。

这符合非凸 GW objective 的预期，并说明 Prompt-4 必须检验初始化与 subsampling
稳定性。当前不能把较高 pairwise JS 解释为稳定 specialization。

## 8. ROI feature variance 与 coupling behavior 是否有明显关系？

Prompt-2R 注册风险保持不变：

```text
Spearman(feature variance, geometry stability) = 0.7760
p = 0.0236
```

未改变主 geometry、未重新加权 ROI，也未用这些相关选择候选。探索性结果：

- feature-only 所有 row 完全相同，entropy/max/depth correlation 无定义；
- balanced outer OT 的 variance-vs-expected-depth rho 为 `0.357`，p=`0.385`；
- sr-GW/sr-FGW 的 variance-vs-expected-depth 多在 `0.65–0.72`；
- beta=0.25 无 coverage 的该相关为 `0.720`，p=`0.044`；
- beta=0.25、lambda=0.01 中 variance-vs-row-entropy rho=`-0.733`，p=`0.039`，
  但该配置的获胜初始化未完全收敛。

这里只有 8 个 ROI，分布含大量 exact ties，并且 repeat/SNR reliability 不可用。
这些结果只登记为 Prompt-4 sensitivity risk，不支持方法选择或层级解释。

## 9. faithful MOT-style inner+outer reproduction 是否可行？

```text
MOT_STYLE_BASELINE_STATUS = NOT_CURRENTLY_REPRODUCIBLE
```

当前 FGW cache 只保存固定维 ROI tokens，不保存每个 ROI 的 per-voxel stimulus
response functions。CLIP cache 保存 mean-patch 的 1,024-dimensional layer
embedding，没有声明一个用于 inner transport 的 per-unit/per-patch domain。Stimulus
ID 对齐是完整的，但 faithful inner brain/model domains 没有同时保存在缓存中。

原 NSD tar 中确实存在 voxel response，可在未来单独构建 per-voxel cache；同时
还需要明确并缓存 CLIP inner unit/patch response、定义 inner costs/regularization，
分别验证 inner/outer transports。本 Prompt 只记录该计划，没有实现 MOT-style
baseline，也没有把 `balanced_outer_ot` 标签为 MOT。本文不声称 global
region-layer transport 本身具有新颖性。

## POT reference 与 tests

当前 `brainx` 环境没有 Python Optimal Transport：

```text
POT_AVAILABLE = false
status = NOT_RUN_DEPENDENCY_UNAVAILABLE
```

因此没有声称自定义 solver 与 POT 数值一致，也没有把 coverage solver 称为
standard POT solver。替代验证由 synthetic tests 完成，覆盖：

1. 非负 Euclidean simplex projection；
2. 每行严格为 1/8；
3. 总 transport mass 为 1；
4. target marginal 总和为 1；
5. balanced target 严格为 1/6；
6. uniform target 最小化 coverage；
7. feature routing 恢复显然低 cost assignment；
8. matched structures 的 GW objective 更低；
9. controlled projected solver 收敛；
10. multi-init 只按 discovery objective 选择；
11. zero entries 不产生 coverage NaN；
12. calibration deterministic；
13. POT comparison 在 POT 可用时才启用，本环境明确 skip。

直接测试结果：

```text
POT reference test skipped: package 'ot' is unavailable
FGW correspondence tests passed
```

## 输出合同与停止点

`correspondence_discovery_v1/` 下有 14 个 candidate directory。每个都包含：

```text
transport_plan.npy
transport_plan.csv
target_marginal.json
coupling_diagnostics.json
optimization_trace.csv
solver_config.json
all_initializations.npz
```

每个 solver config 都保存 M、C_brain、C_clip、feature/geometry config、manifest 和
calibration hashes。每条 trace 包含 raw/normalized feature/GW、coverage、entropy、
total objective、gradient norm、step norm、source feasibility、objective improvement
和 iteration。

完整性检查确认 14 个计划及其 provenance 均符合合同，且输出树中不存在
`best_transport_plan.npy`。Feature-cost 输出约 40 KiB，候选输出约 6.8 MiB。

Prompt 3 到此停止：没有查看 offline validation/test，没有使用 caption metrics，
没有声明最佳科学方法、层级或稳定 specialization，也没有进入 Stage-2。
