# NeuroRoute-FGW 跨被试锁定离线测试复现（Prompt 5B）

## 结论

`CROSS_SUBJECT_STATUS = REPLICATION_SUPPORTED`

在完全锁定的科学配置下，两个独立复现被试 `subj02` 和 `subj05` 都达到
`EXPLORATORY_PASS`。两者相对冻结非结构基线的标准化 GW 损失均下降，配对
测试子采样的 95% 置信区间全低于零，并且均通过三类预注册结构零模型检验。
这一结果发生在 `beta = 0.5`、`lambda_cov = 0`、`entropy = 0` 下，没有进行
被试特异的科学超参数重调。

三个冻结最终计划中，同名 ROI 行的平均 cosine similarity 为 `0.8333`，显著
高于独立打乱每个被试 ROI 行标签所得的预注册零分布；对应的平均 JS divergence
为 `0.1155`，显著低于零分布。两项单侧经验检验均为
`p = 1 / 10001 = 0.00009999`。

这里的“支持复现”仅指：该离线数据划分、冻结 srFGW 配置和三个被试范围内的
结构拟合优势与 ROI 行对应关系。它不证明皮层层级、唯一的脑区—网络层映射、
因果关系，也不代表 Stage-2 或 caption 指标已经改善。

## 锁定协议与数据纪律

- 开发被试：`subj01`，只读取已有冻结计划和测试摘要，没有重跑或修改。
- 独立复现被试：`subj02`、`subj05`。
- 不合格被试：`subj07`，保持 `INCONCLUSIVE`，没有加载其 offline-test，
  没有生成最终传输计划，也没有将其计作 PASS 或 FAIL。
- ROI 固定顺序：`V1, V2, V3, hV4, FFA, EBA, PPA, OPA`。
- CLIP 层固定顺序：`L4, L8, L12, L16, L20, L24`；跨被试比较没有重排层。
- 方法固定为 `sr_fgw`；`beta = 0.5`、`lambda_cov = 0`、`entropy = 0`。
- 脑几何固定为 `geometry_v2`；特征代价固定为 5-fold OOF 多输出 ridge
  ROI→CLIP-layer prediction cost，`ridge alpha = 100`。
- 最终训练集为 `offline_discovery ∪ offline_validation`。计划冻结并写入
  `final_plan_frozen.marker` 后，程序才允许加载该被试 offline-test 特征。
- `subj02` 使用 Prompt-5A 冻结初始化 `feature_informed/0`；`subj05` 使用
  `random/1013`。没有重新筛选初始化。
- 每个被试只拟合一个主要最终计划。非结构基线也使用 Prompt-5A 已冻结配置。
- 没有运行 Stage-2、修改 Perceiver 或执行 caption evaluation。

Prompt-5B 预测试协议的 SHA256 为
`69cd3692938320bf6e072a562b07857c816d968c017dafdd7b82881d48e7516c`。

## 被试级离线测试结果

下表中的 Δ 均为 `srFGW − frozen non-structural baseline`，越低越好。结构零模型
列给出三类检验中最大的 p 值。

| 被试 | 角色 | N train / test | FGW composite | ΔFeature norm | ΔGW norm | ΔComposite | 最大结构零模型 p | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| subj01 | development | 6847 / 1712 | 0.550821 | -0.083539 | -0.249509 | -0.166524 | 0.004975 | EXPLORATORY_PASS |
| subj02 | replication | 6847 / 1712 | 0.531931 | -0.080662 | -0.230371 | -0.155516 | 0.004975 | EXPLORATORY_PASS |
| subj05 | replication | 6545 / 1636 | 0.484610 | -0.077059 | -0.334309 | -0.205684 | 0.009950 | EXPLORATORY_PASS |

两个复现被试的 ΔGW 中位数为 `-0.282340`，范围为
`[-0.334309, -0.230371]`，方向一致。按预注册要求，没有对仅三个被试进行推断性
t 检验。

### subj02

- 原始/标准化 feature loss：`0.143242 / 0.779363`。
- 原始/标准化 GW loss：`0.055339 / 0.284499`。
- composite：`0.531931`。
- ΔFeature norm：`-0.080662`。
- ΔGW norm：`-0.230371`。
- ΔComposite：`-0.155516`。
- 200 次、80% 无放回配对子采样的 ΔGW norm 95% CI：
  `[-0.238050, -0.224399]`。
- brain identity、CLIP identity、independent-ROI stimulus 三类结构零模型的
  p 值均为 `0.004975`。
- 所有冻结 PASS 条件均满足，因此复现 subj01 的离线结构优势。

最终计划 SHA256：
`a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0`。

### subj05

- 原始/标准化 feature loss：`0.137662 / 0.777797`。
- 原始/标准化 GW loss：`0.033497 / 0.191423`。
- composite：`0.484610`。
- ΔFeature norm：`-0.077059`。
- ΔGW norm：`-0.334309`。
- ΔComposite：`-0.205684`。
- 200 次、80% 无放回配对子采样的 ΔGW norm 95% CI：
  `[-0.344164, -0.326825]`。
- 三类结构零模型 p 值依次为 `0.004975`、`0.009950`、`0.004975`。
- 所有冻结 PASS 条件均满足，因此复现 subj01 的离线结构优势。

最终计划 SHA256：
`a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0`。

`subj02` 和 `subj05` 的哈希相同不是复用同一文件：两个被试分别使用自己的
训练表征、几何、特征代价、损失尺度和冻结初始化独立拟合，最终收敛到了相同的
离散 one-hot 计划。

## 最终计划与跨被试相似性

冻结计划的描述性 preferred layer 如下。这只是耦合解的描述，不应解释为皮层
解剖距离或皮层层级。

| ROI | subj01 | subj02 | subj05 |
|---|---:|---:|---:|
| V1 | L4 | L4 | L4 |
| V2 | L4 | L4 | L4 |
| V3 | L12 | L12 | L12 |
| hV4 | L16 | L16 | L16 |
| FFA | L24 | L20 | L20 |
| EBA | L24 | L24 | L24 |
| PPA | L20 | L24 | L24 |
| OPA | L24 | L24 | L24 |

所有计划的平均行熵约为零，说明当前无熵正则解落在 one-hot 顶点。三个计划的
target marginal 都是：`L4=0.25, L8=0, L12=0.125, L16=0.125,
L20=0.125, L24=0.375`；target entropy 为 `1.494175`，effective target layer
count 为 `4.455660`。

| 被试对 | flatten Spearman | flatten Pearson | Frobenius | cosine | target marginal JS | preferred-layer agreement |
|---|---:|---:|---:|---:|---:|---:|
| S1–S2 | 0.700 | 0.700 | 2.000 | 0.750 | 0.000 | 0.750 |
| S1–S5 | 0.700 | 0.700 | 2.000 | 0.750 | 0.000 | 0.750 |
| S2–S5 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |

因此，S1/S2/S5 的最终计划具有明显但非完全一致的跨被试相似性；差异集中在
FFA 与 PPA，而不是 target marginal 总量。

### 同名 ROI 行置换零模型

预注册零模型对每个被试独立打乱 ROI 行标签，保持计划值、目标层使用量、稀疏度、
行熵和 CLIP 层身份不变，共 `10,000` 次，seed=`52001`。

| 统计量 | 实际值 | 零分布均值 | 零分布 95% 区间 | 单侧经验 p |
|---|---:|---:|---:|---:|
| 同名 ROI 平均 cosine | 0.833333 | 0.250679 | [0.125000, 0.458333] | 0.00009999 |
| 同名 ROI 平均 JS | 0.115525 | 0.519390 | [0.375455, 0.606504] | 0.00009999 |

同名 ROI 行确实比任意 ROI 标签匹配更相似；分析始终保持原生 CLIP 层顺序，
没有为提高相似度而优化或置换层身份。

## ROI 级稳定性

| ROI | 平均 pairwise cosine | 平均 pairwise JS | preferred-layer agreement | 初始化歧义 | 解释 |
|---|---:|---:|---:|---|---|
| V1 | 1.000 | 0.000 | 1.000 | 未标记 | 高稳定 |
| V2 | 1.000 | 0.000 | 1.000 | 未标记 | 高稳定 |
| V3 | 1.000 | 0.000 | 1.000 | 未标记 | 高稳定 |
| hV4 | 1.000 | 0.000 | 1.000 | 未标记 | 高稳定 |
| FFA | 0.333 | 0.462 | 0.333 | subj01、subj02 | 低稳定/有歧义 |
| EBA | 1.000 | 0.000 | 1.000 | 未标记 | 高稳定 |
| PPA | 0.333 | 0.462 | 0.333 | subj02 | 低稳定/有歧义 |
| OPA | 1.000 | 0.000 | 1.000 | 未标记；保留 subj01 先验警示 | 跨被试高稳定 |

最稳定的是 `V1, V2, V3, hV4, EBA, OPA`。`FFA` 和 `PPA` 在 S1 与复现被试
间发生 L20/L24 交换，是当前最明确的歧义来源。`OPA` 虽在 subj01 早期初始化
分析中被列入 FFA/PPA/OPA 警示集合，但冻结最终计划在三个被试间一致，因此不应
强行将其结论写成不稳定。

## 几何的 final-train / test 一致性

| 被试 | brain Spearman | CLIP Spearman | brain Frobenius | CLIP Frobenius |
|---|---:|---:|---:|---:|
| subj01 | 0.992337 | 0.996429 | 0.112333 | 0.076230 |
| subj02 | 0.996716 | 1.000000 | 0.109962 | 0.075414 |
| subj05 | 0.993432 | 0.996429 | 0.089822 | 0.064658 |

这说明冻结协议估计出的 ROI 表征几何关系矩阵和 CLIP 表征几何关系矩阵在
final-train 与独立 test 间高度一致。这里的脑矩阵仍然只能称作“ROI
representational geometry relation matrix”，不能称作解剖距离。

## Feature-pairing 次级检验

直接 feature-pairing 证据不具有跨被试一致性：subj01 offline-test 的次级检验
为 `p=0.047619`，而 subj02、subj05 均为 `p=1.0`；Prompt-5A validation 中该项
也没有形成稳定支持。按照预注册，该检验不参与 PASS 决策，也没有据此改变 beta。

因此目前支持的是结构几何项带来的优势，而不是一个已稳定复现的直接线性
ROI→CLIP feature compatibility 信号。

## 共同 held-out 刺激审计

`SHARED_STIMULUS_ANALYSIS = INSUFFICIENT_HELDOUT_SHARED_STIMULI`

- subj01 / subj02 / subj05 的 test 数分别为 `1712 / 1712 / 1636`。
- 三者共同 held-out stimulus 数为 `0`，每一对的交集也为 `0`。
- 预注册最低要求为 `200`，因此没有降低阈值，也没有执行共同刺激特征分析。
- 这是次级分析，不改变 `CROSS_SUBJECT_STATUS`。

## Prompt 要求的 13 个明确回答

1. **subj02 是否复现 subj01 的离线结构优势？** 是，`EXPLORATORY_PASS`。
2. **subj05 是否复现？** 是，`EXPLORATORY_PASS`。
3. **subj07 是否在 INCONCLUSIVE 后保持未触碰？** 是。目录哈希前后均为
   `da01eb821a47fba089cfc8e0813890f6044a9995bfef2c6a6a5d0c2d61166963`；
   offline-test 未加载，最终计划未生成。
4. **srFGW 是否在独立复现被试上优于冻结非结构基线？** 是。S2/S5 的
   ΔComposite 与 ΔGW 均小于零，且 ΔGW 子采样 CI 全小于零。
5. **该结果是否发生在 lambda_cov=0？** 是，严格为 `lambda_cov=0`。
6. **S1/S2/S5 最终计划是否相似？** 是，但非完全一致。S2/S5 完全相同；
   S1 与两者各有 6/8 ROI 的 preferred layer 一致。
7. **同名 ROI 是否比 ROI 标签置换零模型更相似？** 是，cosine 和 JS 检验
   均 `p=0.00009999`。
8. **哪些 ROI 最稳定？** `V1, V2, V3, hV4, EBA, OPA`。
9. **哪些 ROI 仍有歧义？** `FFA`、`PPA` 最明显；`OPA` 保留早期初始化警示，
   但其冻结最终计划跨被试稳定。
10. **feature-pairing 证据是否跨被试一致？** 否；S1 p=0.047619，S2/S5 p=1.0。
11. **CROSS_SUBJECT_STATUS 是什么？** `REPLICATION_SUPPORTED`。
12. **现在支持哪些主张？** 在当前离线冻结协议与被试范围内，`beta=0.5`、
    `lambda_cov=0` 的 srFGW 结构拟合优势可在两个独立被试复现；同名 ROI 行的
    耦合分布也超过预注册行置换零模型；早期视觉 ROI 以及 EBA/OPA 的描述性
    preferred-layer 对应在三个冻结计划中稳定。
13. **哪些主张仍不受支持？** 皮层或 CLIP 的生物学层级、唯一/可识别的
    ROI-layer 真映射、FFA/PPA 的稳定对应、稳定的直接 feature-pairing 信号、
    因果或解剖解释、跨 NSD 总体推广、subj07 的测试复现、共同刺激上的跨被试
    泛化、Stage-2/Perceiver/caption 性能提升，均未由 Prompt-5B 证明。

## 输出与审计文件

- `fgw_outputs/subj02/correspondence_offline_test_replication_v1/`
- `fgw_outputs/subj05/correspondence_offline_test_replication_v1/`
- `fgw_outputs/cross_subject_fgw_replication_v1/subject_replication_summary.csv`
- `fgw_outputs/cross_subject_fgw_replication_v1/subject_replication_summary.json`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_plan_similarity.csv`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_roi_similarity.csv`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_plan_null_summary.json`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_plan_null_values.npz`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_geometry_summary.csv`
- `fgw_outputs/cross_subject_fgw_replication_v1/shared_stimulus_analysis.json`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_replication_status.json`
- `fgw_outputs/cross_subject_fgw_replication_v1/cross_subject_provenance.json`

subj05 的第一次长任务在 offline-test 解锁前、feature-pairing null 计算过程中被
会话中断。中断目录没有冻结 marker、测试几何或测试代价。恢复流程保留并逐字节
核验了已完成且唯一的主要/基线最终计划，没有再次拟合或替换初始化，只重新计算
尚未保存的预测试 null 后再执行原锁定驱动程序。恢复证据保存在
`interruption_recovery_provenance.json`，中断快照保留于
`correspondence_offline_test_replication_v1_interrupted_pretest_20260820/`。

本报告到此停止；没有启动 Stage-2 或任何下游实验。
