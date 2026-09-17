# UMBRAE 扩展版本演进与实验索引

本文把仓库中的实验按“研究问题”而不是按文件生成时间分类，便于后续阅读、复现和继续开发。这里的“版本”指一次有独立假设、实现和结论的主要实验阶段；随机种子、debug run 和同一方法的单独 subject run 不重复列为版本。

现有目录没有物理搬迁。原因是配置、脚本和 provenance 中已经记录了这些路径，直接移动会破坏复现。本文作为稳定的逻辑索引，原始机器可读结果继续保留在各自的 `*_outputs` 目录。

## 一、快速结论

如果只想知道下一步从哪里继续：

| 目标 | 当前最值得保留的版本 | 结论 |
| --- | --- | --- |
| 原始系统复现 | B1 UMBRAE compatibility reproduction | 单样本端到端链路已验证 |
| Stage-1 表征对齐 | N3 soft routing + projector | 四个受试者上稳定优于 uniform/L24，但没有 ROI 层级解释 |
| 严格 Shikra caption | N6 UMBRAE + uniform ROI multi-layer | S1 caption 指标最好；ROI 应补充而非替代 UMBRAE |
| 参数高效校准 | P5 shared LoRA | 仅 0.168% BrainX 参数，表征指标改善且语义保持 |
| 冻结表征接口适配 | P9 token-wise adapter | 验证集 caption/grounding 有小幅一致改善，P10 没有进一步提升 |
| 离线 ROI-layer correspondence | F6 cross-subject srFGW | S2/S5 复现通过，但 FFA/PPA 仍有歧义 |
| FGW Stage-2 | F8 gamma=0.25 | 比 uniform 提高 val CIDEr，但 row-shuffled 更好，尚不能证明对应关系有因果价值 |
| 细粒度 pRF 表征 | P11-B | 正确 pRF 分组未胜过匹配随机对照，当前方案不应继续扩规模 |

总的判断是：多层 ROI 信息有用，但现有可学习 routing、粗 ROI 结构和 pRF 分组还没有稳定证明解剖特异性。当前最可靠的下游方案仍是保留 UMBRAE 全局 token，再加入覆盖稳定的 ROI 多层信息。

## 二、命名和结果路径

| 系列 | 研究主线 | 主要位置 |
| --- | --- | --- |
| B | 原始 UMBRAE 与复现 | `README.md`、`REPRODUCTION.md`、`umbrae/evaluation/` |
| N | NeuroRoute 基础、Stage-1、Stage-2 | `umbrae/models/`、`umbrae/stage1_outputs/`、`umbrae/umbrae_neuroroute_outputs/` |
| P | Dual-Branch P3–P11 递进实验 | `umbrae/dual_branch_outputs/`、`umbrae/configs/dual_branch/` |
| F | FGW correspondence 与 Stage-2 | `umbrae/fgw_outputs/`、`umbrae/docs/FGW_*.md` |

状态含义：`推荐基线` 表示适合继续开发；`有效但有限` 表示假设部分成立；`诊断完成` 表示主要价值是排除机制；`未验证` 表示不能把结果解释成原假设成立。

---

## 三、B 系列：原始 UMBRAE 与复现

### B0 — 上游 UMBRAE 基线

- **方法**：BrainX/BrainXS 将 fMRI 编码为 256 个 1024 维视觉 token，经 Shikra `mm_projector` 映射后替换 prompt 中的 `<im_patch>`。
- **作用**：提供所有扩展实验的全局语义基线和严格下游接口。
- **效果**：在 P7 锁定测试中，基线 CIDEr 0.6064、BLEU-4 0.1903、ROUGE-L 0.4362、grounding accuracy 0.1976。
- **不足**：没有显式 ROI 结构；下载和依赖原始版本不够稳定；本仓库后续实验使用的评测协议并不都与论文完全相同。
- **建议**：任何新结构都必须同时与 B0 和容量匹配的随机结构对照比较。

### B1 — Compatibility reproduction

- **方法**：固定 Python 3.10、PyTorch 2.3/CUDA 12.1、Transformers 4.31、Accelerate 0.20.3 和 NumPy `<2`；增加 `--max_samples`、`--max_new_tokens` 以支持有界 smoke test。
- **改进**：解决新驱动/新依赖与旧 Shikra/Transformers 代码不兼容的问题，同时保留原默认行为。
- **效果**：S1 单样本完成 fMRI → BrainX → projector → Shikra，生成 caption “A group of people cooking in a kitchen”；checkpoint 无 missing/unexpected key。
- **不足**：只验证了一个样本，不等于完整 982 样本 benchmark；仍依赖本地 NSD 和大模型权重。
- **入口**：[REPRODUCTION.md](REPRODUCTION.md)。

---

## 四、N 系列：NeuroRoute 路由版本

### N0 — ROI/CLIP 基础设施

- **方法**：把官方 ROI atlas 映射到 `nsdgeneral.npy` 的 voxel 顺序；建立 V1/V2/V3/hV4/FFA/EBA/PPA/OPA 八个 ROI；增加 CLIP L4/8/12/16/20/24 layer bank 和 ROI tokenizer。
- **改进**：从整脑单一路径扩展到可审计的 subject-specific ROI token 和多层视觉教师。
- **效果**：S1/S2/S5/S7 voxel-order 全部通过真实样本验证；真实 tokenizer shape `[B,3,V] -> [B,8,D]` 通过。
- **不足**：RSC 与 `nsdgeneral` 无交集，被正确排除；结论只适用于当前 mask，不能外推 wholebrain。
- **入口**：[ROI 集定义](umbrae/docs/NEUROROUTE_ROI_SET_V1.md)、`roi_indices/`。

### N1 — 初始 soft routing 与 anti-collapse

- **方法**：比较 soft、uniform、random、hard ROI-to-CLIP-layer routing，并扫描 temperature 和 balance 正则。
- **改进**：首次让每个 ROI 自适应混合多层 CLIP target。
- **效果**：无 projector 时 soft val alignment loss 1.1603，优于 uniform 1.2515、random 1.2687、hard 1.2966；balance 可提高熵、降低最大权重。
- **不足**：所有 ROI 强烈偏向 L24；balance 越强 alignment 越差；没有得到可信的 ROI-specific hierarchy。
- **判断**：`有效但有限`。soft mixture 有优化价值，但不能作神经层级解释。

### N2 — BrainToCLIPProjector

- **方法**：在 raw ROI token 与 CLIP target 之间加入 MLP projector，router query 仍使用 raw token。
- **改进**：把 voxel 聚合/ROI identity 与 CLIP 坐标校准解耦。
- **效果**：S1 loss 从约 1.x 降至 0.1134；soft 继续优于 uniform 0.1403、single-L24 0.1477、random 0.1510、hard 0.1619。
- **不足**：训练延长到 10 epochs 后仍是 L24-biased mixture，说明缺少 ROI hierarchy 不是训练时间不足。
- **判断**：`推荐组件`。projector 是后续 Stage-1 的必要表示桥梁。

### N3 — 四受试者 Stage-1 soft routing

- **方法**：固定八 ROI、六 CLIP 层、projector 和训练协议，分别训练 S1/S2/S5/S7。
- **改进**：把单被试结果扩展为跨被试一致性检查。
- **效果**：soft 平均 val loss 0.1028，uniform 0.1286，single-L24 0.1406；四名受试者中 soft 相对 uniform 均改善约 19.5%–20.5%。
- **不足**：early/high-level ROI expected depth 几乎无差异；只证明自适应多层混合，不证明脑区层级。
- **判断**：`Stage-1 推荐基线`。
- **入口**：[Stage-1 routing results](umbrae/docs/STAGE1_ROUTING_RESULTS.md)。

### N4 — Retrieval 与 contrastive 版本

- **方法**：用 mean-pooled brain ROI 与 CLIP L24 做 B2I/I2B retrieval，并加入权重 0.01/0.03/0.05 的 InfoNCE。
- **改进**：避免只用 alignment loss 评价样本判别能力。
- **效果**：contrastive 0.05 的四受试者平均 B2I R@10 为 9.92%、median rank 94.63，优于 alignment-only soft 的 5.75%/120.88。
- **不足**：alignment component 恶化约 5.5%–7.7%；single-L24 仍有最佳 B2I R@1 和多数 I2B 指标；早期 attention+routed 版本接近随机且训练不足。
- **判断**：`任务权衡`。若目标是 B2I 中高召回可用 0.05；不能称为全面提升。
- **入口**：[Stage-1 retrieval results](umbrae/docs/STAGE1_RETRIEVAL_RESULTS.md)。

### N5 — Generic-prefix Stage-2

- **方法**：在严格 Shikra bridge 尚未完成时，以通用 `inputs_embeds` 比较 L24-only、routed-only、concat-soft、concat-uniform、concat-single-L24。
- **改进**：快速验证 ROI token 是否包含 caption 下游信号。
- **效果**：concat-soft 有最低 val loss 2.1543；routed-only 有最高 BLEU-4 0.0359/CIDEr 0.2123。
- **不足**：不是 `<im_patch>` 协议；只跑 S1、三 epochs、单参考 caption，不能与论文或严格版本直接比较。
- **判断**：`原型验证`，不作为最终架构。

### N6 — Strict UMBRAE/Shikra integration

- **方法**：`256 UMBRAE + 8 ROI -> 256-query Perceiver -> 256×4096`，严格原位替换 256 个 `<im_patch>`；冻结 UMBRAE、NeuroRoute 和 Shikra，只训练 fusion adapter。
- **改进**：恢复与原始 Shikra token 接口一致的公平下游比较。
- **效果**：S1 上 UMBRAE+uniform 达到 BLEU-4 0.0912、ROUGE-L 0.3072、CIDEr 0.7582，优于 UMBRAE-only 的 0.0712/0.2949/0.5942；soft 也明显优于 UMBRAE-only 和 single-L24。
- **不足**：NeuroRoute-only CIDEr 仅 0.2460；uniform 全指标优于 learned soft；caption 只验证 S1，且不是论文零训练推理设置。
- **判断**：`当前 strict caption 推荐基线 = UMBRAE + uniform ROI multi-layer`。
- **入口**：[strict integration results](umbrae/docs/UMBRAE_NEUROROUTE_RESULTS_S1.md)。

### N7 — Structured routing

- **方法**：测试 `(1-alpha)·uniform + alpha·soft`（alpha 0.1/0.3/0.5）及 temperature 2.0。
- **改进**：尝试在 uniform 的覆盖稳定性与 soft 的自适应性之间折中。
- **效果**：alpha 0.3 的 val loss 与 uniform 近似，但 caption 明显更差；所有 alpha/tau 版本均未超过 uniform。
- **不足**：alpha 越大，覆盖越低、深层偏置越强、caption 越差；ROI diversity 与 early/high gap 仍接近零。
- **判断**：`负结果/诊断完成`。继续单纯调温度或线性混合价值较低。

---

## 五、P 系列：Dual-Branch P3–P11

### P3 — Stage-A semantic UOT

- **方法**：用 semantic unbalanced OT 训练 real ROI 与容量匹配的 random partition。
- **改进**：引入严格相同初始化和随机结构对照，检验粗 ROI 是否提供真实结构优势。
- **效果**：两者都从初始化显著学习表征。
- **不足**：random 的 RSA/retrieval 优于 real；`STRUCTURE_SIGNAL=NEGATIVE`。
- **判断**：`需要调整`。优化成功不等于结构假设成立。

### P3R — ROI relational objective

- **方法**：改为 ROI-wise relational loss + global contrastive loss，减少 P3 的 token collapse。
- **改进**：显式保持 ROI 间关系和样本判别性。
- **效果**：real mean ROI Spearman 0.498、R@10 0.683，显著优于旧 P3；ROI specialization signal 转正。
- **不足**：random 仍更强（mean ROI Spearman 0.667、R@10 0.757），粗解剖 ROI 信号仍弱。
- **判断**：`表示学习有效，结构证据为负`。

### P4 — Gated fusion calibration

- **方法**：冻结 BrainX 和 structural branch，只训练 gated fusion；比较 real 与 random structure。
- **改进**：小残差融合并强制 semantic preservation。
- **效果**：real RSA 从 0.8528 提高到 0.8635，random 到 0.8663；preservation cosine 约 0.9985。
- **不足**：random 增益更大；结构互补性为负。
- **判断**：`fusion 有效，解剖结构证据弱`。

### P5 — Shared LoRA-only calibration

- **方法**：不使用 structural/fusion，仅训练 shared LoRA，245,760 参数，占 BrainX 0.168%。
- **改进**：建立参数高效且语义保持的校准基线。
- **效果**：RSA 0.8676、R@1 0.9967、base-LoRA cosine 0.9983，超过 B0 和 P4 real fusion 的 RSA。
- **不足**：P7 下游 caption/grounding 反而下降，说明 representation metric 与 decoder utility 不一致。
- **判断**：`推荐校准基线`，但必须做下游验证。

### P6 — Full Stage-C

- **方法**：LoRA + gated structural fusion，比较 real/random partition。
- **改进**：组合 P4 与 P5，保持整体表示接近基线。
- **效果**：real RSA 0.8745、random 0.8752；相对 B0 均提升约 0.022，R@1 保持 0.9967。
- **不足**：random 略高于 real，coarse structure signal `INCONCLUSIVE`；组合交互并非简单相加。
- **判断**：`完整模型有效，但不能归因于真实 ROI`。

### P7 — Locked downstream validation

- **方法**：在锁定 test 上同时比较 UMBRAE、LoRA、Full Real、Full Random 的 caption 和 grounding。
- **改进**：第一次检验高 RSA 是否转化为真实下游收益。
- **效果**：Full Real 相对 Full Random 在 CIDEr、BLEU-4、ROUGE-L 和 grounding accuracy 都略高；但绝对效应小。
- **不足**：UMBRAE 基线仍有更高的总体 CIDEr/BLEU/ROUGE；LoRA 下游为负；未计算不确定性/显著性。
- **判断**：`DOWNSTREAM_UTILITY=MIXED`，解剖信号只能称为 task-specific directional evidence。

### P8 — Calibration-strength diagnosis

- **方法**：在 UMBRAE 与 Full Real/Random 间做表示插值，并检查 projector 接口、hard retrieval 和样本级结果。
- **改进**：把“结构没用”和“校准过强破坏 decoder 接口”区分开。
- **效果**：Full Real alpha=0.25 是 sweet spot，CIDEr 0.6109、BLEU-4 0.1910、grounding 0.1984，优于 alpha=1 并保持/略超基线效用。
- **不足**：real 与 random 仍接近，解剖证据只具方向性。
- **判断**：`OVER_CALIBRATION`。后续应优先做 preservation-constrained interface calibration。

### P8.5 — Distribution-bias diagnosis

- **方法**：用 train-only 统计拟合 mean、channel affine、WCT 修正。
- **改进**：检验失败是否仅来自均值、方差或协方差分布偏移。
- **效果**：修正能改善部分 hard retrieval，但 affine/WCT 大幅损害 caption 和 grounding；alpha=0.25 明显更好。
- **不足**：边缘统计匹配不保持 decoder interface；Full Real effective rank 4.86，远低于 CLIP 57.30。
- **判断**：`DISTRIBUTION_BIAS_HYPOTHESIS=NOT_SUPPORTED`，主要问题是过校准而非简单分布偏移。

### P8.75 — Exact CLIP-token oracle

- **方法**：GT image → frozen CLIP token → frozen projector/Shikra，测量当前 decoder 的理论上限。
- **改进**：区分 decoder ceiling 与 brain-to-visual interface gap。
- **效果**：oracle CIDEr 1.6357、grounding 0.5234，远高于 UMBRAE 的 0.6064/0.1976。
- **不足**：oracle 不是脑解码模型，也不可部署；只用于机制诊断。
- **判断**：`decoder headroom 很大，主要瓶颈是 brain token compatibility`。

### P9 — Token-wise pre-projector adapter

- **方法**：冻结 BrainX/P6/Shikra，只训练 527,617 参数的 residual MLP adapter 和 gate。
- **改进**：直接优化 pre-projector 接口，同时要求 checkpoint 不损害 caption 和 grounding。
- **效果**：UMBRAE validation CIDEr 0.8776→0.8925、grounding 0.2369→0.2405；P6 也小幅改善；仅恢复约 1.1% projected gap。
- **不足**：P6 相对 UMBRAE 的可用信息优势仍弱；结果仅 validation，未跑 post-hoc test。
- **判断**：`推荐的冻结表征接口适配器`；simple token-wise adapter 已足够作为下一版基线。

### P10 — Token-mixing adapter

- **方法**：一层 8-head MHSA + FFN + 双 gate，训练 4.73M 参数，在 256 token 间混合信息。
- **改进**：检验 decoder 需要的信息是否存在但分布在错误 token 位置。
- **效果**：注意力确实使用大量 off-diagonal mass 且未 collapse。
- **不足**：UMBRAE 无 downstream-compatible checkpoint；P6 的 CIDEr/grounding 均不如 P9；参数更多但没有联合改进。
- **判断**：`FROZEN_REPRESENTATION_READOUT_LIMIT=REACHED`。问题更像缺少细粒度信息，不是 token 排列错误。

### P11-A — Fine-grained spatial audit

- **方法**：审计现有 categorical pRF/retinotopy、CLIP patch 几何、voxel-to-patch affinity 和候选细粒度单元。
- **改进**：在训练前先验证空间资产，不伪造连续 pRF center/sigma。
- **效果**：确认 categorical fallback 可构建，但不足以称为 Gaussian pRF-grounded token。
- **不足**：初始本地数据缺少连续 polar angle/eccentricity/size/R²，阶段被正确阻断。
- **判断**：`审计通过，训练暂缓`。

### P11-A.5 — Continuous pRF asset recovery

- **方法**：下载官方 subj01 func1pt8mm pRF angle/eccentricity/size/R²/exponent，并通过同网格 C-order 精确映射到 `nsdgeneral`。
- **改进**：不做插值/表面近邻，保留 CSS effective-size 与 raw sigma 的语义区别。
- **效果**：4656/4657 early-visual voxels 有完整 pRF；mapping validation 通过；R²≥10.1 时保留 3956 voxels。
- **不足**：仅 subj01；阈值和 K 仍需在训练前锁定。
- **判断**：`BLOCKER_RESOLVED`，允许进入 P11-B0。

### P11-B0 — pRF-grounded unit construction

- **方法**：用 Gaussian voxel→16×16 CLIP patch affinity，比较 2 个质量策略、2 个 feature set 和 K32/48/64/96/128；在不看下游的前提下选型。
- **改进**：建立 real、within-ROI×hemisphere random 和 teacher-shuffle 三种容量匹配控制。
- **效果**：锁定 `R²≥10.1 + XY + K64`（64 retinotopic + 4 high-level token）；所有单位至少 12 voxels。
- **不足**：center sampling 近似 patch integral；高 K 产生小单位；high-level ROI 没有 pRF teacher；mapping 不能跨被试直接复用。
- **判断**：`训练接口准备完成`。

### P11-B — pRF-grounded fine representation

- **方法**：real 与随机分组使用相同 unit size、teacher W、初始化和 30-epoch protocol，优化 local cosine/MSE、unit relation 和 cross-sample InfoNCE。
- **改进**：首次严格检验正确 pRF voxel membership 是否优于容量匹配随机结构。
- **效果**：两者 local cosine 都约 0.625；real relation Spearman 0.4931 高于 random 0.4706。
- **不足**：real localization MRR 更低，sample MRR 显著更低；locality gap 均为负，出现 semantic leakage/global collapse；关键指标未胜 random。
- **判断**：`NO_RELIABLE_PRF_STRUCTURE_ADVANTAGE`。当前 fine-grained representation 未验证，应先重审 local encoder 或 fMRI 信息上限。

---

## 六、F 系列：FGW correspondence 与 Stage-2

### F0 — FGW implementation audit 与 leakage-safe cache

- **方法**：审计现有 NeuroRoute 路径，建立固定 discovery/validation/offline-test 划分和 projector 前 ROI representation cache。
- **改进**：把 correspondence discovery 与 protected downstream/test 数据隔离，保留 hashes 和 provenance。
- **效果**：形成后续所有 FGW 版本的可复现输入合同。
- **不足**：cache 对 fMRI repeats 先平均，不能做 repeat-based measurement reliability。
- **入口**：[实现审计](umbrae/docs/NEUROROUTE_FGW_IMPLEMENTATION_AUDIT.md)、[cache 报告](umbrae/docs/FGW_REPRESENTATION_CACHE_S1.md)。

### F1 — ROI geometry reliability/stability V2

- **方法**：用独立 stimulus 子集、逐 ROI permutation null 和 projector 前 token 检验 8×8 ROI representational relation matrix。
- **改进**：V2 修正了旧版 bootstrap/null 与“measurement reliability”的过度表述。
- **效果**：S1 sampling stability 通过，`GEOMETRY_STATUS=GO`，允许继续离线 FGW。
- **不足**：`REPEAT_RELIABILITY_STATUS=NOT_AVAILABLE`；geometry 不是解剖距离，不能推断皮层层级；只有 8 ROI，variance confound 需谨慎。

### F2 — S1 srFGW discovery

- **方法**：比较 feature-only、balanced OT、GW、srFGW 及 coverage variants；在 discovery split 上扫描 beta/lambda/初始化。
- **改进**：联合 ROI geometry 与跨模态 feature cost，显式检查 transport collapse、coverage artifact 和初始化敏感性。
- **效果**：发现结构项能改变 coupling，并形成待 validation 的候选。
- **不足**：MOT-style inner+outer baseline 因缺少 per-voxel stimulus feature 无法忠实复现；POT reference 在环境中不可用；此阶段仅 discovery。
- **入口**：[S1 discovery](umbrae/docs/FGW_CORRESPONDENCE_DISCOVERY_S1.md)。

### F3 — S1 locked validation

- **方法**：在独立 validation 上比较候选、结构 null、feature pairing null、near-optimal plan 和 variance sensitivity。
- **改进**：预注册阈值后锁定 `sr_fgw, beta=0.5, lambda_cov=0, entropy=0, seed=1006`。
- **效果**：`READY_FOR_OFFLINE_TEST`；整体可识别性通过，使用五个 CLIP 层，真实 geometry 优于多个结构 baseline/null。
- **不足**：FFA/PPA/OPA 行存在局部最优歧义；仍是 subj01 exploratory，未授权 Stage-2。
- **入口**：[S1 validation](umbrae/docs/FGW_CORRESPONDENCE_VALIDATION_S1_V1.md)。

### F4 — S1 one-shot offline test

- **方法**：锁定配置后在 discovery+validation refit，再一次性读取 1712 个 offline-test stimulus。
- **改进**：避免看 test 调参，并加入三类结构 null、subsampling 和 feature-pairing 检查。
- **效果**：`EXPLORATORY_PASS`；三类结构 null 均 p=0.00498。
- **不足**：feature-pairing 只有边界 p=0.04762 且与 validation p=1.0 不一致；仍不能跨被试外推。
- **入口**：[S1 offline test](umbrae/docs/FGW_CORRESPONDENCE_OFFLINE_TEST_S1_V1.md)。

### F5 — Cross-subject locked validation

- **方法**：不救援超参数地把 S1 锁定配置迁移到 S2/S5/S7。
- **改进**：将 correspondence 从单被试探索转为真正的独立复现筛选。
- **效果**：S2、S5 `READY_FOR_OFFLINE_TEST`；S7 `INCONCLUSIVE`。
- **不足**：不是所有被试都通过；只允许 S2/S5 进入后续一次性 test。
- **入口**：[cross-subject validation](umbrae/docs/FGW_CROSS_SUBJECT_VALIDATION_V1.md)。

### F6 — Cross-subject offline replication

- **方法**：对 S2/S5 使用完全锁定的 beta=0.5、lambda=0 srFGW，并与 baseline、结构 null、ROI row permutation 比较。
- **改进**：检验 S1 的离线结构优势能否在独立 subject 和不相交 test stimuli 上复现。
- **效果**：S2/S5 都为 `EXPLORATORY_PASS`，总状态 `REPLICATION_SUPPORTED`；V1/V2/V3/hV4/EBA/OPA 较稳定。
- **不足**：FFA/PPA 的 L20/L24 发生交换；feature-pairing 证据不一致；三个被试无共同 held-out stimulus，因此无法做共同刺激特征分析。
- **判断**：`当前离线 FGW 推荐版本`，但结论仍限于当前 protocol/subject 范围。

### F7 — FGW Stage-2 attention prior implementation

- **方法**：把冻结 transport plan 转成第一层 cross-attention 的 log prior；实现 uniform、FGW 和 row-shuffled FGW，且不增加 trainable correspondence 参数。
- **改进**：严格验证 plan hash/axis/row mass、动态 token slice、gamma=0 等价和 leakage；保持 48-token architecture 完全一致。
- **效果**：实现与测试通过；S1/S2/S5 plan allow-list 固化，subj07 被拒绝。
- **不足**：此版本只完成工程实现，没有训练或科学结果。
- **入口**：[Stage-2 implementation](umbrae/docs/FGW_STAGE2_IMPLEMENTATION_V1.md)。

### F8 — FGW Stage-2 S1 development

- **方法**：S1 上比较 uniform、FGW gamma 0.25/0.5/1.0 和预注册 row-shuffled control；所有架构、参数量与数据相同。
- **改进**：只用 validation multi-reference CIDEr 选择 gamma，protected test 保持封存。
- **效果**：锁定 gamma=0.25，val CIDEr 0.4222，高于 uniform 0.4004；状态 `READY_FOR_REPLICATION`。
- **不足**：row-shuffled gamma=0.25 更高（CIDEr 0.4429、BLEU-4 0.1818），说明当前提升不能归因于正确 ROI correspondence；FGW 的 BLEU-4 也低于 uniform。
- **判断**：`可进入 S2/S5 复现，但 correspondence 因果价值未验证`。
- **入口**：[S1 Stage-2 development](umbrae/docs/FGW_STAGE2_S1_DEVELOPMENT_V1.md)。

---

## 七、推荐的后续开发顺序

1. **先固定任务目标**：alignment、retrieval、caption 和 grounding 的最优版本不同，不再用单一 RSA/alignment 指标选择所有下游 checkpoint。
2. **caption 主线从 N6 开始**：以 `UMBRAE + uniform ROI multi-layer` 为强基线，再测试真正带约束的 subject/ROI-conditioned 小残差；不要继续只扫 temperature/alpha。
3. **接口主线保留 P9**：新表示必须同时比较无 adapter、P9 token-wise adapter 和 B0，避免把接口不匹配误判为表示无效。
4. **FGW 主线先复现 F8**：在 S2/S5 固定 gamma=0.25，并保留 row-shuffled control；只有正确 plan 稳定胜 shuffled 才进入 protected test。
5. **pRF 主线暂停扩大 P11-B**：先解决 semantic leakage/locality collapse，可尝试限制全局语义通路、局部对比负样本和 patch-area integral；在 real 胜 matched random 前不做下游大模型训练。
6. **补齐不确定性**：P7 的小幅 real-random 差异需要 bootstrap/多 seed；strict caption 需要跨 S2/S5/S7 和多参考评测。

## 八、阅读顺序

建议按以下顺序阅读：

1. 本文；
2. [NeuroRoute 全实验总报告](umbrae/docs/NEUROROUTE_ALL_EXPERIMENTS_SUMMARY.md)；
3. [Dual-Branch protocol](umbrae/docs/DUAL_BRANCH_PROTOCOL_V1.md)；
4. P7 → P8 → P9 → P10 → P11 的对应 report；
5. [FGW implementation audit](umbrae/docs/NEUROROUTE_FGW_IMPLEMENTATION_AUDIT.md)；
6. F1 → F8 的对应 `FGW_*.md` 报告；
7. 需要精确数值时再查看相应 output 目录中的 JSON/CSV。

本文总结的是仓库现有证据，不把工程通过、单被试趋势或 oracle 结果提升为更强的科学结论。后续新增版本时，应沿用“方法 / 改进 / 效果 / 不足 / 判断”的格式追加，并链接锁定配置与机器可读结果。
