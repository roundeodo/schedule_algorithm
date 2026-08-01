# OLMoE 特征分布调度器升级固定计划

> 状态：已完成的推导/证据日志，不是当前 RTL 规范。当前唯一控制规范为
> `BOUNDED_DISTILLED_SCHEDULER.md`，入口为
> `scheduler_rtl_distilled_policy.py`。

## 固定目标

本任务只处理以下目标，完成前不更换方法论或训练目标：

1. 对固定的 65 条 OLMoE 风格 Top-2 投影分布，获得显式 DMA lane 四阶段模型中的全局最优调度路径。
2. 每条“最优”声明必须同时满足：合法 history 可重放、`LB = UB`、`proven_optimal = true`。
3. 在同一批输入上评估现有 HW-v2 和实验性 top4+bottom2 Python 镜像，不使用 mirror 小于合法 reference 的 ghost-prefetch 结果作为改进证据。
4. 按窗口、候选 action、控制状态、评分器四层顺序定位损失并升级 Python 镜像。
5. 最终 Python 调度器应达到已证明的最优周期，同时保持可观察窗口、候选数量和评分计算有界且适合后续 RTL 实现。
6. 本阶段不修改 RTL。

## 固定数据与基线

### 分布集合

- 基础 43 条：`results/policy_search/olmoe_top2_projection_cases_v3.json`
  - SHA-256: `235f96035e25d27bc8fa3352804cd8f85861e05018d8b6d1e043fb7bbe85e038`
- minimum-positive-load=2 补充 22 条：`results/policy_search/olmoe_top2_projection_min2_supplement_v1.json`
  - SHA-256: `ee24240623e6dd2e7c1561e21956137d9f937a61f605836903b27c577e87bab8`

在 65 条全部得到最优证书前，不向验收集合增加新分布。新分布只能在算法冻结后的独立泛化测试中加入。

### 当前证据状态

- 已证明全局最优：65/65（base 43/43，minimum-positive-load=2 补充集 22/22）。
- 43 条 base 最终 proof 为
  `results/policy_search/olmoe_top2_projection_base_optimal_v1.json`，SHA-256
  `cb48043439176b7641f244b2320c2ed2a692bfbc7891959577fac0cf04756f5b`。
  与 22 条 min2 proof 合并审计时，65/65 均满足 `proven_optimal=true`、
  `history_replay_valid=true`、`LB=UB`，每条 history 均重新在显式 DMA
  reference 中 replay 为记录的 makespan。
- `olmoe_median_joint_constraint_profile` 已由 `hot_tail` target scout 在
  root semantic group `PAIR(12,22)` 找到 120-tick 可行 history；32 个显式
  DMA action 独立 replay 为 1,351,680 cc，正好等于 certified LB=120，故为
  全局最优。永久 campaign 为
  `results/policy_search/olmoe_median_target120_hottail_campaign_v1.json`，
  SHA-256 `d0b96758505f616a67dc10a54bb12eccdaf333788f3286320318b35c745ac9eb`；
  原 median `dma` campaign 已停止，避免继续搜索同一目标。
- `olmoe_iqr25_joint_constraint_profile` 由 `hot_tail` 在 root semantic group
  `PAIR(4,15)` 的 60 秒 checkpoint 上继续搜索，累计约 509 秒找到 126-tick
  history；37 个显式 DMA action 独立 replay 为 1,419,264 cc，等于
  certified LB=126，故为全局最优。永久 campaign 为
  `results/policy_search/olmoe_iqr_target126_hottail_campaign_v1.json`，
  SHA-256 `dd008b22cebb3e258f9ada7fe148b5cb1d3493b5ecd285b526be8ebf8dc6b84b`。
  IQR 的正式 `dma` campaign 和所有 scout 均已停止。
- `olmoe_grid_hot6_8x_dual_a43_le2_42` 已由构造式 DMA-saturation
  history 从 `[129,130]` 闭合为 `[129,129]`；完整 reference replay 通过。
- 热点/冷尾精确负载类别 DP、合法 four-stage lowering 和一次 SPLIT
  构造已将证书集合从 5/65 提升到 56/65；所有 canonical histories 均已
  再次独立 replay。构造器只提供 UB，只有命中独立 certified LB 时才转为证书。
- 双 slot inter-block S4PF 构造又将证书集合提升到 57/65：
  `olmoe_min2_grid_hot6_8x_dual_a29_le2_49` 先运行 `PAIR(7+7)`，
  分别预取 16-token anchor 和 2-token 冷 expert，再用 `PAIR(16+2)` 进入
  冷尾覆盖段，完整 reference replay 将 `[108,111]` 闭合为 `[108,108]`。
- 下界前沿精确搜索将证书集合从 57/65 提升到 63/65。它从 certified LB
  所在的首个合法时间格点开始判定，并用 depth 顺序找到以下异步合法路径：
  - `olmoe_min2_grid_hot6_8x_dual_a33_le2_46`：`[117,120] -> [117,117]`；
  - `olmoe_min2_grid_hot8_12x_triple_a33_le2_46`：`[108,111] -> [108,108]`；
  - `olmoe_min2_grid_hot12_14x_dual_a38_le2_44`：`[117,120] -> [117,117]`；
  - `olmoe_dual16_median_cold_profile`：`[120,123] -> [120,120]`；
  - `olmoe_grid_hot6_8x_dual_a33_le2_46`：`[120,123] -> [120,120]`；
  - `olmoe_grid_hot12_14x_dual_a38_le2_44`：`[117,120] -> [117,117]`。
  这些路径证明同步 block 构造器会漏掉合法异步交错，因此它继续只作为 UB
  构造器，不得转化为 admissible LB。
- 已新增完整 target-feasibility 搜索；10 组双专家和 12 组三专家
  小问题上，`OPT-1 cc` 均完整判定不可达，`OPT` 均找到可重放
  history，与完整 anytime exact search 一致。
- target 搜索的安全 future-work dominance、生成器 bound 开关、多源 OPEN 和
  checkpoint continuation 均已做 A/B：小型 feasible/infeasible verdict 与一次性
  完整搜索一致；逐 expansion checkpoint 恢复后的累计 expansions 也完全一致。
- 将完整 state LB 提前放入每个 action 生成的实验虽减少 action 数，但 30 秒
  expansions 从 59 降到 8，已从默认路径撤下；当前默认保持关闭。
- 300 秒串行精确搜索与不含显式冷尾窗口的 W64 seed beam 均未产生相应收益；
  后者已停止，不再作为全量方案。
- 在剩余 case 上补充的合法 `block`/`block_cache` W64 beam 将
  `olmoe_median_joint_constraint_profile` 的 replay-valid UB 从 123 收紧为
  122 ticks；两个排序均得到 122，随后又以 W128 `block_cache` 复核但未进一步
  改善。该结果只作为构造性 UB，不参与最优性证明。单 case 永久证据为
  `results/policy_search/olmoe_median_block_w64_v1.json`，SHA-256 为
  `20a7331c2a576edcce616835c74583e798f5236183d849af9f69c973c0e3d3c5`；
  30 个 action 已独立在显式 DMA reference 中 replay 为 1,374,208 cc。
- 当前 proof 输入与最终结果：
  - `results/policy_search/olmoe_top2_projection_base_best_known_v1.json`
    SHA-256: `b6f68691971b80a84fd0b5794048b720ccdab9e5b052226d65943eadedab548c`
    （保留为两个 root campaign 的不可变 prior，不再作为最终 43-case proof）
  - `results/policy_search/olmoe_top2_projection_base_optimal_v1.json`
    SHA-256: `cb48043439176b7641f244b2320c2ed2a692bfbc7891959577fac0cf04756f5b`
    （最终 43/43 base proof）
  - `results/policy_search/olmoe_top2_projection_min2_best_known_v1.json`
    SHA-256: `e835bb3e2f1b1beb20bf0cfc0d4a827e8eff75ce6b9417123e709953a2d1a9ea`
- 阶段 B 的统一 65-case 严格基线为
  `results/policy_search/olmoe_65_stageb_baseline_v1.json`，SHA-256
  `9f102f80ffd50ec51206b9a9ed7862ba7c2078b2a2841247414c34181666642a`。
  独立审计重新 replay 130 条 HW-v2/top4+bottom2 合法 histories 和 58 条
  可用 oracle histories；输入/源码哈希一致，min2 22 条与独立先导运行逐字段
  零漂移。HW-v2/top4+bottom2 合法累计 gap 分别为 1,359/1,361 ticks，
  平均每 case 20.91/20.94；逐 case top4+bottom2 胜 24、HW-v2 胜 22、
  相等 19，两者均仅 1/65 达到最优。
  W32 mirror oracle 在 49/65 上改善 HW-v2、selection gain 合计 170 ticks，
  但仍累计高于最优 1,113 ticks且仅 1/65 命中；65/65 都发生 beam pruning，
  所以没有 candidate-sufficiency 证书。7 条 oracle trace 无法 post-hoc
  lowering，进一步确认正式阶段 D 必须使用显式 DMA candidate oracle。

### 模型冻结点

- `four_stage_scheduler.py`
  - SHA-256: `8509def6b05b19910258240d8af3dd95bb95e239b91395bf8fb59dd700050fc2`
  - 完整 reference 的默认 action 语义未改变；新增 exact target 的时间格点、
    future-work dominance、多源初始 OPEN、排序开关和可序列化 checkpoint。
  - exact target 现可选 `candidate_window=(TOP,BOTTOM)`：先从完整 remaining
    构造 TOP+BOTTOM+resident 的有序可见集合，再在该集合内运行完整
    future-distinct 生成；已物化且仍在 remaining 的 prefetch resident 保持可见。
    `None`、显式 `None`、覆盖全部 expert 的窗口在小型回归中 history、verdict、
    expansions/generated 等逐项一致；检查点恢复与窗口配置不一致会被拒绝。
- `scheduler_hw_fixed_policy.py`
  - SHA-256: `591971d4296e687f3f6f5d7ef9872af72f8b877d9a8c042f6b7f55ac958f67d4`
  - 仅新增不改变选择逻辑的逐 transition trace API；与修改前冻结的
    65-case 报告逐 case 比较，65/65 makespan 完全一致。
- `scheduler_top4_bottom2_policy.py`
  - SHA-256: `7ebc093f83f1740fd3ce946bd6639158290652a77200a2cfb2fedb546bedb9c3`
- `construct_olmoe_block_schedules.py`
  - SHA-256: `61fa0d5b22e13df39400457c4ec4690d7c9821a6a1b636b014d24fad157e19a9`
- `prove_top4_bottom2_directed.py`
  - SHA-256: `eabbc5b2ac8c4858d96ad94242520b3a78a3b2200002c24d5d83369d629a79fa`
  - 显式 DMA lowering 新增 `stage_only` 保守路径：保持 mirror 每轮选择的
    expert-count 与 cluster 发射序列，但不把 ghost prefetch 强行物化；具体
    shape、开始时间和 DMA lane 仍由完整 reference 合法 action 生成器决定。
    因为逐轮贪心 shape 会在后续死路，`stage_only` 使用沿固定发射序列的
    fingerprint 去重回溯。其终态必须通过独立 history replay；搜索失败只表示
    lowering 未找到，不能证明该发射序列不可实现。
- `run_target_root_branches.py`
  - SHA-256: `10da93e77102cb67ce1df2cfeb37d3a8910e46b3c26f3a8eba20376fb74d5358`
  - root children 与后续 exact search 使用同一个窗口过滤；受限窗口 campaign
    必须显式给出已证明 `LB=UB` 的 optimum target，manifest 固化窗口、目标来源、
    prior/reference 哈希。相同配置可恢复，不同窗口复用 work-dir 会被拒绝。
  - 修复 `--group-index` 与 `--repeat-until-complete` 帮助语义和参数校验互相
    矛盾的问题；指定 root group 现在可从 checkpoint 自动续跑，找到可行 history
    或该指定 group 精确耗尽时结束。`counts=[4,2]` 的指定 group 小型回归在
    2 expansions 后找到 6-tick history；这项修改只影响 campaign 控制，不改变
    reference action、剪枝或 state fingerprint。
- `merge_top4_bottom2_proofs.py`
  - SHA-256: `29b477242d5110901c6a7bbea47738544cb29c6f34d095fa9cbe4b621bd9db7f`
  - 可直接消费 complete root-campaign：验证 prior/reference SHA、case/counts、
    root-group 完整性和 history replay；feasible frontier 生成 `LB=UB` 证书，
    全 root groups exhaustive 时只将 LB 推进一个时间量子。普通 proof payload
    的 65-case merge 回归关键字段完全一致；两个当前 incomplete campaign 均被
    拒绝；feasible 与 exhaustive 两条转换路径的单元回归通过。
  - 默认仍拒绝受限窗口 campaign，因其失败不能推进完整 reference 的全局 LB。
    仅在显式使用 `--allow-restricted-witness` 时，允许从已完成且找到 target 的
    受限 campaign 中提取构造性 witness；提取前验证全局 prior 已 `LB=UB`、
    target 等于全局 optimum、物理 replay、逐 action 原始窗口可见性和 terminal。
    该开关不能把受限失败转成全局 proof。base43 与 min2-22 合并回归仍得到
    65/65 optimal。
- `run_window_exact_priority_campaign.py`
  - SHA-256: `ac49cc0f4f136bbd8446f0a0ac9a6feffaf8b56f714f648fbc39c79027b21388`
  - 对单个 case/window 的 root semantic groups 进行 round-robin 受限 exact
    target 搜索；PAIR、SINGLE、平衡 SPLIT 等只影响根分支尝试顺序，不改变
    admissible pruning 或窗口合法 action 集。
  - 每次 invocation 使用独立时间片并保存 checkpoint，已尝试但未闭合的 group
    不会阻塞其他 root groups；`--max-invocations` 只用于短测或有界 campaign，
    设为 0 才表示继续到找到 witness 或全部 root groups 穷尽。
  - manifest 固化 prior/audit/reference/runner/self 哈希和窗口配置；活动 campaign
    启动后不得修改脚本或跨窗口复用 work-dir。
- `analyze_scheduler_hw_candidate_oracle.py`
  - SHA-256: `cb3c5ffcf5c6f24e44a9eac7b6e16da4ba44b5641d5fee149c39d9b51f853553`
  - 阶段 B 入口现在分别记录 HW-v2 与 top4+bottom2 的当前 mirror 值和
    late/proactive/stage-only 显式 DMA lowering；只有独立 replay 通过的 legal 字段用于
    物理性能比较，mirror 只作为候选图和 ghost-prefetch 诊断。报告同时保存
    完整合法 actions、lowering 记录，以及相对一条已保存最优 history 的首个
    expert/family 与物理 action 分歧；该分歧只作诊断，不排除另一条等周期最优路径。
  - candidate beam 也保留被选 transition trace 并进行显式 DMA lowering/replay。
    W1 基础 43 条与改动前逐 case 比较，makespan、expanded、generated、deduplicated、
    pruned、terminal 和 peak-retained 统计均完全一致。一个 mirror-optimal trace 的
    lowering 失败不能排除另一条等 mirror-cost trace，因此这仍是候选诊断，不替代
    阶段 D 的受限合法 candidate oracle。
  - `--strict-directed` 强制 suite/reference 名称及 distribution 一致、65/65
    reference proved、所有 reference histories 重新 replay、HW-v2 与
    top4+bottom2 两条已部署策略 trace 均可合法 lowering，并拒绝任何低于已证明
    optimum 的 legal 结果。mirror candidate-oracle lowering 是可用
    `--strict-oracle-lowering` 单独开启的诊断门禁，不属于正式基线门禁。
    当前 65 条同输入和 replay 检查全部通过；严格门禁按预期拒绝尚未闭合的 2 条。
  - 正式命令使用 `--expected-directed-cases 65`；43-case 单集被范围门禁拒绝，
    65-case 全集随后被 proof 门禁拒绝当前 2 条未证明输入，验证顺序符合预期。
  - `--directed-case` 只用于小型端到端回归。已对一条 proved case 完整运行
    strict JSON 管线：reference=129，HW-v2 mirror/legal=153/153，top4+bottom2
    mirror/legal=153/153，W1 oracle mirror/legal=153/153；三条合法 histories
    分别保存 24/26/24 个 actions，首个 expert-family 分歧均在 round 0，报告
    自身源码 SHA 与文件实际 SHA 一致。正式 65-case 命令不使用该过滤器。
  - 旧的逐轮贪心 lowering 曾在
    `olmoe_min2_grid_hot8_12x_dual_a29_le2_49` 的最后一轮错误失败；加入
    `stage_only` 回溯后严格单例完整通过。reference optimum=111，HW-v2
    mirror/legal=126/135（stage-only），top4+bottom2 mirror/legal=114/115
    （late），所有 legal histories 均再次 replay。这里 stage-only 的 135 只是
    保持发射序列的保守合法上界，不宣称精确复现 HW-v2 的抽象 prefetch timing。
    `stage_only` 仅在 late/proactive 均失败时启用，不与更忠实的两种 lowering
    按 makespan 竞争。
- `evaluate_directed_window_grid.py`
  - SHA-256: `c1c87bc0ad643dd6527c98782828f306ee3bef4423bdf96f95be497a8f60f906`
  - 修复窄窗口找到“合法但未达到 target”的 history 后过早停止其他 beam
    width/rank-mode 的错误。回归用旧 unproven target 复现首个 W8 completion
    只能达到 123、target 为 121 的情况；修复后继续执行 cache probe，而不是
    把第一个次优合法 history 当作搜索完成。
  - window fragment 配置现在包含 case-input、target-proof、额外 witness 和相关
    Python 源码的 SHA-256；显式 work-dir 具有 manifest，一旦内容不一致或发现
    无 manifest 的旧 fragments 就拒绝复用。相同配置连续运行两次的 manifest
    回归通过。
  - `--require-all-proven-targets` 为正式窗口评估提供硬门禁；当前完整 base
    输入因 2 条未证明而按预期拒绝，单条已证明子集按预期通过。
  - 正式 base/min2 命令分别使用 `--expected-cases 43/22`；范围门禁先于
    proof 门禁，43-case 输入冒充 65 被拒绝，正确范围随后因 2 条未证明被拒绝。
  - probe summary 保存 expanded/generated 以估算正式运行成本。
  - 新增 `--direct-only`，只审计已保存、可重放的等周期最优 histories，不启动
    heuristic probe；不可见项保持 `unresolved`，不会误报窗口不足。
  - 新增 `--prior-window-audit`：只接受完整的 direct-only audit，逐一验证
    case-input、global proof、reference 与全部 witness 文件 SHA-256；只复用
    `proved_sufficient_direct`，旧 unresolved、heuristic 和 exact verdict 均不会
    被提升。新 witness 只重放旧 audit 未覆盖的窗口。单 case 正常复用通过，
    min2 audit 冒充 base 输入被 case-input hash 门禁拒绝；22 条新增 exact
    witnesses 的四窗口汇总从超过 10 分钟的重复 relabel 降为 1.49 秒。
    复用链可递归展开每一层 prior audit，并验证每层 audit、自身 target 和
    witness 文件 SHA；循环或同一路径哈希冲突均拒绝。两层 top8 审计回归通过。
    已被 audit 引用的 witness 文件从此不可覆盖；新增命中使用增量 witness，
    再由下一层 audit 组合，保持全部历史结论可重放。
  - 新增 `--exact-campaign`：验证 campaign 完整性、reference SHA、case/counts、
    optimum target、root-group exhaustion，以及 feasible history 的物理 replay 和
    逐 action 窗口可见性。报告严格区分 direct/exact sufficient、exact insufficient
    和 unresolved。用 `counts=[4,2]` 的小型用例验证：保存的 PAIR history 对 top1
    不可见，但受限 exact 找到等周期 SPLIT/SINGLE history，汇总正确标记为 exact
    sufficient。

后续如果修改 reference 模型，必须重新验证已保存 history；如果只修改待升级策略，reference 证书保持独立。

## 固定执行阶段

### 阶段 A：全局最优路径与证书

1. 使用更宽的完整 action beam 搜索收紧合法 UB，不把 beam 结果称为最优。
2. 独立保存每个 case 的 history、UB、LB、termination 和 source hash。
3. 多次结果只按以下安全规则合并：`UB = min(UB_i)`、`LB = max(LB_i)`。
4. 当且仅当合并后 `LB = UB` 时生成最优证书。
5. 对未闭合 case 采用精确分支定界、按目标周期的可行性判定、强化 admissible LB 或根分支并行；任何启发式裁剪不得参与最优证明。

完成条件：65/65 条 `history_replay_valid = true` 且 `proven_optimal = true`。

### 阶段 B：现有调度器同输入基线

逐 case 保存：

- 全局最优 ticks；
- HW-v2 Python 镜像 ticks；
- top4+bottom2 Python 镜像 ticks；
- 首个与某条最优 history 分歧的 round；
- 合法 lowering/replay 状态。

完成条件：所有比较使用相同分布、相同初始 cache、相同 tick 定义和相同 DMA 约束。

#### 负载均分与合法组合消融（2026-07-29）

为避免把计算下界误当成可执行调度，新增
`analyze_olmoe_balance_requirement.py` 和
`evaluate_olmoe_simple_balance_policy.py`，分别输出：

- `results/policy_search/olmoe_65_balance_requirement_v1.json`；
- `results/policy_search/olmoe_65_simple_balance_policy_v1.json`；
- `results/policy_search/olmoe_65_distribution_catalog_v1.md`。

三个文件 SHA-256 依次为
`8bee4a64d801e55b6097c3e94b3e6f425f82e958adba32f2b6ccbe96061a65b`、
`6f938f958e65d91249ab9126891534b2a0212b4e223d8784f73dfc7385ef1ed8`、
`2fd6f9b91bf39d0da5e2b0e3698343cf2aeb2d840ba63a61ff6c772b80010c20`；
生成脚本 SHA-256 依次为
`b17074d3cfe76bfdc2d095b0e0b7e5a798cb8f0d84aa1bcd30207e229c738c8f`、
`520974547403f8ca2297867a35a807e39ab04ace6b3ca812c8aba7303c0bfca0`。

结果严格区分计算估计与可重放物理调度：

- 原始 token LPT 在 61/65 条达到最佳完整-expert 计算分区；改用
  `ceil(tokens/2)` 计算块后为 65/65。两者都没有生成 DMA 合法 history。
- 最佳计算分区只在 57/65 条等于显式 DMA 四阶段物理最优；其余 8 条存在
  3--48 ticks 的物理代价。最大 48 ticks 来自 uniform control 中大量 1-token
  expert，不能据此外推所有 OLMoE 热点分布。
- 最优 histories 只有 9/65 条把原始 token 完全均分；token 差中位数为 6，
  最大为 28。相反，47/65 条的两个 cluster 最终完成时刻完全相同，其余也都
  相差不超过 3 ticks。因此优化目标是完成时间而不是 token 数相等。
- 两个没有 continuation scorer、没有显式 free-prefetch 的合法简单策略均已
  65/65 replay：同步边界固定取 top-top 只有 1/65 最优，平均比最优慢 22.70%；
  固定取 hot-cold 也只有 1/65 最优，平均慢 26.32%。现有 HW-v2 平均慢
  17.71%，说明动态组合已经比两种固定规则有效，但仍不是最终方案。

因此后续候选器必须表达“热点 anchor + 可变数量中冷 expert 的连续填充”，评分器
必须比较预计完成时刻和暴露的 DMA stall；不能退化成 raw-token 均分，也不能固定
每轮 top-top 或 hot-cold 配对。

### 阶段 C：窗口充分性

#### 2026-07-29 正式实验合同

1. 正式比较且只比较 `top5+bottom1`、`top8+bottom2`、`top8+bottom4`、
   `top8+bottom8`。此前的 top4/top6 等窗口结果只作为候选路线的历史证据，
   不再消耗新的 exact 搜索预算。
2. 停止继续投入 `top4+bottom2` exact 搜索。已有结果、日志与 checkpoint 保留为
   baseline，状态必须写成 `unresolved`，不得写成 insufficient。停止点已归档为
   `results/policy_search/window_exact/olmoe_observed_top4_bottom2_stopped_checkpoint_v1.{json,tar.zst}`。
3. 窗口阶段只改变可观察 expert 集合，不改变完整显式 DMA action 语义、全局
   optimum target、候选上限或评分器。对等负载 expert 只允许全局一致的 ID
   重命名，重命名后的完整 history 必须重新 replay。
4. 覆盖结论分成三类：
   - `sufficient`：存在逐 action 窗口可见、物理 replay 合法且 makespan 等于
     全局已证明 optimum 的完整 history；
   - `insufficient`：对应 optimum target 下，全部合法 root semantic groups
     均被受限 exact search 穷尽；
   - `unresolved`：除此之外的任何情况，包括 beam 未命中、超时和只检查了
     某一条保存 history。
5. 先复用窄窗口的合法 witness：`top5+bottom1` witness 可作为三个 top8 窗口
   的候选证据；`top8+bottom2` witness 可复用于 bottom4/bottom8；
   `top8+bottom4` witness 可复用于 bottom8。每次复用仍必须独立执行窗口可见性
   检查和物理 replay。
6. 当前构造性覆盖下界为（2026-07-29，top5 priority-exact 已提取 26 条
   base witnesses，top8 全局根定向搜索新增 2 条 witnesses 的审计快照）：

   | 窗口 | base 43 | min2 22 | 合计 65 | unresolved |
   |---|---:|---:|---:|---:|
   | top5+bottom1 | 38 | 22 | 60 | 5 |
   | top8+bottom2 | 41 | 22 | 63 | 2 |
   | top8+bottom4 | 41 | 22 | 63 | 2 |
   | top8+bottom8 | 41 | 22 | 63 | 2 |

   这些数字只是已经找到的 constructive witnesses，不是最终覆盖率，也不能据此
   提前选择最大窗口。三个 top8 W32 `block_cache` 筛查均已完整结束：bottom2、
   bottom4、bottom8 分别新增 2、1、2 条 beam witnesses；它们随后都被更窄的
   top5 exact witnesses 覆盖。输出 SHA-256 依次为
   `28b6f7a3c43508d225c4cae19915495022af709ebaa7220c72e137732e6a4270`、
   `0f8c71981785ea7e1eb906be20c97c856c8d63505a78faeea213884c1427dee8`、
   `5a8fe51046666e966ec422ad88b9c3333de1b43704fd51098be6e1a1dbf4b907`。
   统一四窗口审计为
   `results/policy_search/window_exact/olmoe_base_stagec_cross_witness_audit_v1.json`
   （SHA-256
   `2595c779a8347fef3f8bb3500f9fb1f83e1be39564d7457d5d49660972e72cf9`）。
   当前 top5 exact 继续处理 5 条 unresolved；新增的
   `hot6 triple a38` witness 已通过独立可见性检查和物理 replay。
   最新 delta witness 和 audit 为
   `olmoe_t5b1_base_priority_exact_witnesses_delta4_v1.json`、
   `olmoe_base_stagec_top5_post_delta4_audit_v1.json`（SHA-256 分别为
   `42634ce4921a194c1a35d6d072de30b89f1812fc2a1598a5ab0970db75798256`、
   `46f6760bc5d9946d917130983d6b24b1db54e8765d5198ea688bbb79cc1f2e54`）。
   top8+bottom2 第一轮与两个全局根定向续跑累计新增 5 条；新增的
   `hot8 dual a38`、`hot6 quad a43` witnesses 已独立 replay 到三个 top8
   窗口，使其均达到 63/65，仅 `observed_ranked` 与 `hot6 dual a38` 未决。
   增量 witness 和最新 top8 审计为
   `olmoe_t8b2_globalroot_exact_witnesses_delta2_v1.json`、
   `olmoe_base_stagec_top8_post_globalroot_delta2_audit_v1.json`（SHA-256 分别为
   `e95f1e8deaf9664ec1cf9484f63fc24ce2be7d2defb3a93940db987f5e66bb2d`、
   `1825ca5de393c6db78473bd3c27fd8bc631f60a0f4b3209bc6a885b15dc0d814`）。
   bottom2 新 witness 向 bottom4/bottom8 复用；只有 bottom2 精确不足的 case
   才继续运行更大 bottom 窗口。
   top8 主 campaign 已从旧审计冻结的 4 条 case 收缩为真正未覆盖的
   `observed_ranked` 与 `hot6 dual a38`；两者原有 root-group fragments
   与 checkpoints 已在相同 case manifest 下复用，不重复消耗已覆盖 case
   的 exact-search 预算。
   新增 `extract_window_exact_witnesses.py`（SHA-256
   `fc6290fb9f3c27c2c8b53d61d1cd35b678bf167c5060ea80c4955bc5470c26ff`）
   作为唯一 exact-fragment 抽取门禁：它同时校验 case manifest、
   fragment SHA、`LB=UB` target、逐 action 窗口可见性和完整显式 DMA
   replay。已用 delta4 回归，抽取 history 逐 action 与原固化 witness
   完全一致，且可独立覆盖全部四个正式窗口。

下面保留的旧窗口记录是方法开发和 baseline 的审计历史；若与本节正式实验合同
冲突，以本节为准。

当前 63 条已证明 history 的直接构造性 witness 仅覆盖：top4 8/63、top6 12/63、
top8 12/63、top14 26/63、top16 30/63、top32 63/63；对这些特定 histories
加入 bottom2/4/8 未增加覆盖。该统计只反映 offline 搜索所保存的任意一条最优
history，并不证明小窗口不足，也不能据此选择 top32。其作用是确定后续哪些
case/window 必须运行受限 target 搜索。

65 条统一 proof 已固化为
`results/policy_search/olmoe_top2_projection_65_optimal_v1.json`，SHA-256
`b7b9e50aa258d0283fc1327fa47e3f1ce4b853df9e19366a268f0b9ca28dd49e`；reference
加入窗口过滤后，65 条 history 已再次逐条 replay 且均保持 `LB=UB`。结合 median
替代 witness 的直接窗口审计为
`results/policy_search/olmoe_65_direct_window_witness_audit_v1.json`，SHA-256
`63a36466710f3a1ff2d3f187b588365e88455d361d31e9c00cf97de308c9c4c0`。
直接构造覆盖为 top4 8/65、top4+bottom2 19/65、top6+bottom2 29/65、
top8+bottom2 29/65、top32 65/65；这些数字只减少 exact 搜索项，不构成任何
小窗口不足结论。

首个受限 exact campaign 已证明 IQR25 的 `top4+bottom2 @ 126 ticks` 充分。
窗口先于对称去重且 hidden-work budget 加速后的 `PAIR(15,1)` root group 累计
61.7 秒、4,342 expansions、16,228 generated，找到 38-StageAction history
（37 个 consuming action 加 1 个显式 prefetch）；独立物理
replay 为 126 ticks，逐 action 均满足 TOP4+BOTTOM2+resident 可见性，并与全空间
certified `LB=UB=126` 相等。complete campaign 为
`results/policy_search/window_exact/olmoe_iqr25_top4_bottom2_target126_v3.json`，
SHA-256 `366a4d2a701f337d3c9bc15b6fce91490a9fb75571cc872a940e6b1f06990333`；
独立 exact-window 汇总为
`results/policy_search/window_exact/olmoe_iqr25_top4_bottom2_exact_audit_v1.json`，
SHA-256 `730eccebe62392bad1c70ddda6ffc5c8be5c493a99560683fd787e82f2363111`。
这证明原保存 history 的 `PAIR(15,4)` 不可见不代表窗口不足；同一窗口可用 cold
bottom 替代根动作并保持全局最优。加上 direct/median witnesses，TOP4+BOTTOM2
当前已有 20/65 条构造性充分证据；其余 45 条先运行 replay-valid restricted
beam 只寻找 witness，未命中项再进入 accelerated exact，不把 beam miss 当结论。

IQR25 的该受限最优 history 由 1 个根 PAIR、36 个 SINGLE 和 1 个 standalone
PREFETCH 组成；37 个 issued experts 的动态窗口来源为 top0 20 次、bottom1 15 次，
top1/top2/top3 各 1 次。根部是 `PAIR(top0=15,bottom1=1)`，随后 one-idle 在
热点/中间头部与冷 bottom 间交错。这是对“one-idle 至少必须同时提供 head 与
bottom 候选”的构造性证据，但不单独证明其他可见 rank 可以删除；阶段 D 仍用
candidate oracle 做删减。

TOP4+BOTTOM2 的 W32 `block_cache` 初筛在 min2-22 中新增一条独立 witness：
`olmoe_min2_grid_hot8_12x_quad_a29_le2_49` 的 29-expert 分布在受限窗口内找到
108-tick history，独立 replay 等于全局 `LB=UB=108`。永久 witness 为
`results/policy_search/window_exact/olmoe_min2_hot8_quad_top4_bottom2_witness_v1.json`
（SHA-256 `542f02007c8959a7fb6d632f5241c5b179249d5b8bb25e813ba8dc463754925f`），
独立单例窗口审计为
`results/policy_search/window_exact/olmoe_min2_hot8_quad_top4_bottom2_witness_audit_v1.json`
（SHA-256 `069b8afcde830a68c249361cc62d62a30693f108206ce0c368a6b4f773b1ae79`）。
因此在 base W32 筛查完成前，TOP4+BOTTOM2 至少已有 21/65 条构造性最优 witness。

20 组窗口的 direct-only 网格已分别固化为
`results/policy_search/window_exact/olmoe_base_direct_window_grid_v2.json`
（SHA-256 `7de9325dab748ac8ed2704476cc4b4d9b67705ac8ceb5563ca83a9a806adb885`）和
`results/policy_search/window_exact/olmoe_min2_direct_window_grid_v2.json`
（SHA-256 `cea811ffb3809bb60cf6c6eedb9316d3876c93381aeebfe965d948d489472937`）。
其中 min2-22 的保存/替代最优 histories 对 TOP6+BOTTOM2 直接覆盖 19/22，
而 TOP4+BOTTOM2 为 13/22；两者在完整 65-case 上的当前直接或已审计 exact
覆盖分别至少为 29/65 和 21/65。这仍只是构造性覆盖，不是最小窗口结论；
TOP6+BOTTOM2 的 base/min2 W32 restricted screen 已并行启动，未命中项仍须 exact。

median 新得到的原始 120-tick 最优 history 直接需要 top14：它先取热点，随后
交错中间负载和当时位于 rank 13 的 2-token expert。但在完整 reference 中把
每次 `<=2`-token issue 改为当前 bottom expert，并同步更新实际 `ntok` 后，
cluster、shape、DMA 和时间均可保持不变；新 32-action history 独立 replay
仍为 120 ticks，且全程 `top4+bottom2` 可见。该单 case 构造性 witness 为
`results/policy_search/olmoe_median_top4_bottom2_optimal_witness_v1.json`，
SHA-256 `8c339dd5d934cc70f676e668c85e66ad52f8d25ee09e5cd1c6f1b6f9fc927a12`；
窗口 grid 的独立单例门禁确认它直接覆盖 target、无需 restricted search。
这证明“保存的某条最优 history 需要 top14”不能推出该 case 的最小窗口大于
top4+bottom2；后续仍按受限 exact target 搜索确定所有未覆盖 case。

IQR 的 126-tick 保存 history 根轮为 `PAIR(15,4)`，4-token expert 初始
zero-based rank=18；该 history 在已测试窗口中仅 top32 直接可见，且不同负载
不能用 equal-load ID relabel 消除。`top4+bottom2` W64 的 `block_cache`、
`block`、`cache` 三个合法 beam 分别展开约 1.9K 状态、总生成约 1.10M actions，
均未找到 window-valid terminal；输出中的 126 只是窗口外 prior replay bound，
三个 trial 的 `window_history_found=false`。这只把 IQR 标为 restricted-exact
高优先级 case，不证明 top4+bottom2 不足。

流程 pilot：`olmoe_observed_ranked_window_001` 的 optimum 为 129 ticks，保存
history 仅 top32 直接可见。top4+bottom2 的 W8/W32 `block_cache` 分别展开
309/1263 个状态、生成 77,719/73,470 个 actions，top8+bottom2 W32 展开
1,251、生成 87,316；三者均未找到完整窗口合法 history，单次约 20--24 秒。
这些是启发式未命中，不是窗口不足证明；它们说明正式全量 heuristic screen
为分钟到小时级，未覆盖项仍需 checkpointed restricted exact target search。

受限 exact target 的实现冻结点（待当前两个无窗口 campaigns 退出后修改
`four_stage_scheduler.py`）：

1. 先按当前 state 构造 TOP+BOTTOM 可见 eid，并加入已显式预取且仍在 remaining
   的 resident eid；随后才在这个有序子集上运行完整
   `gen_stage_actions`/`gen_prefetch_actions`。窗口选择必须早于等价负载类别的
   representative reduction。
2. 不修改 admissible LB、target capacity、future-work dominance 或 state
   fingerprint。窗口只改变合法 action 集；rank ordering 仍只影响搜索顺序。
3. `candidate_window` 必须进入 `TargetFeasibilityCheckpoint` 配置并参与恢复校验，
   防止不同窗口共用 OPEN/CLOSED。
4. `candidate_window=None` 要在小型 feasible/infeasible 集上与当前实现逐 verdict、
   expansions/generated 对比；各窗口的 beam 命中 history 必须由 restricted exact
   搜索复现；OPEN exhaustion 才能作为对应窗口在 target 上不足的证明。
5. 若进一步按 root semantic groups 分解，root children 也必须用同一可见性过滤，
   且全部受限 root groups 穷尽后才能声明窗口不足。

实现审计曾发现一个已撤销的错误顺序：先对完整 remaining 做等价类别去重、再按
窗口过滤，会在大量相同冷负载时保留中间位置的两个代表并删除真正可见的 bottom
expert。IQR25 根部旧顺序生成的 bottom action 数为 0；修正为“窗口先于对称去重”
后为 595，root semantic groups 从 15 增为 19，并出现合法 `PAIR(15,1)` 与
`SINGLE(1)`。该错误版本的 services 已全部停止，其 checkpoints/logs 已删除，
未作为任何充分或不足证据。无重复负载类别用例的新旧 future-state 集完全一致；
无窗口 exact 回归仍为 210 expansions、1711 generated，确认完整 reference 未漂移。

窗口可见集合会隐藏一部分未参与本轮 action 的 expert，但
`_minimum_cluster_work` 对 expert 可加。当前实现把 hidden experts 的该项常数
从 work/capacity budget 中预扣，使生成器在可见子集内的两项提前剪枝与完整
remaining 条件代数等价；不可加的 composite state LB 不在窗口生成器内部使用，
仍由每个 child 的完整 `within_target` 检查执行。30 个真实 checkpoint states
逐一比较 unbounded-visible 与预扣版本，外层 target 后的 child future-state 集
全部相同；单 state action 数从最多 3,226 降到 0、2,223 降到 147。IQR25
`PAIR(15,1)` 的 30 秒 pilot 达到 2,433 expansions/8,618 generated，而未预扣
版本 300 秒仅 1,485 expansions/1,379,985 generated；pilot checkpoint 已作为
正式 v3 campaign 的起点，所有被替代 v2 checkpoints/logs 已删除。

完成条件：65 cases x 4 windows 的每一项均被归类为 `sufficient` 或
`insufficient`，不存在因 beam miss/timeout 被误判的项。窗口选择不单独按 entry
数决定；只有完成相同固定候选预算和评分器闭环后，才综合选择整体 RTL 代价最低
且性能最好的方案。

#### 2026-07-29 阶段 C 最新冻结状态

- 三个 top8 窗口已经各自取得 65/65 条直接、可重放且等于全局 optimum 的
  history：`top8+bottom2`、`top8+bottom4`、`top8+bottom8` 均为
  `proved_optimal_covered=65`，不存在 unresolved 或 insufficient。统一审计为
  `results/policy_search/window_exact/olmoe_65_stagec_top8_coverage_audit_v1.json`
  （SHA-256
  `fc9b203ec63e17517bc0df8bca4b50767ef89ffd6df83fdafc04df1cae68362c`）。
- `top5+bottom1` 当前为 base 42/43、min2 22/22，即合计 64/65 构造性充分；
  唯一未决项是 `olmoe_observed_ranked_window_001`。后台 priority-exact
  campaign 仍保留 live OPEN/checkpoint；任何 time slice 未命中都继续记作
  `unresolved`，不得据此声明 top5 不足。
- 因 top8 三个窗口已经完成，阶段 D 可以对 `top8+bottom2` 做预备候选器开发，
  但在 top5 最后一项闭合前不得冻结最终窗口。bottom4/bottom8 当前没有带来额外
  最优路径覆盖，因此只有后续同 K 候选器或闭环评分器给出收益时才有保留理由。

### 阶段 D：候选 action 充分性

#### 正式公平比较合同

1. 只对阶段 C 后仍有竞争力的正式窗口进行候选器比较。
2. 每种窗口分别使用完全相同的固定候选预算 `K=16/24/32`；`K` 指每 round
   送入 scorer 的去重后候选上限，不允许大窗口暗中生成更多候选再只报告前 K。
3. 候选充分性仍在显式 DMA reference 图上判断：若固定候选图中存在一条等于
   全局 optimum 的完整路径，则该 case/K/window 为 candidate-sufficient。
   候选 oracle 不负责证明窗口不足，也不评价评分器。
4. 每个候选器同时记录动态和静态复杂度：每 round 原始/去重后 action 数、
   rank/cut/shape/prefetch family 枚举数、最坏比较次数、候选字段位宽、窗口
   entry/双缓冲状态、refill 元数据和必要控制状态位。不得只比较 Python 时间。
5. 选择顺序为：先在同 K 下比较最优路径覆盖，再比较实际策略 gap，最后用
   可综合代价打破性能接近方案的平局。

#### 2026-07-29 第一版 K32 候选器审计结论

`evaluate_olmoe_bounded_candidate_oracle.py`（当前 SHA-256
`b73069b67193d32b65b9176c8a23ca9123fc35c3573bdf04473ac910cc17d829`）已经把
`top8+bottom2` 的每个候选限制为：有界逻辑 load-class template、每 template
最多三个局部物理 profile，并在 family quota/fill 后严格截断为 K。它不把 target
注入候选生成或排序。

对 65 条已保存 top8+bottom2 最优 history 的逐 transition 审计为
`results/policy_search/olmoe_t8b2_k32_candidate_direct_witness_audit_v1.json`
（SHA-256
`26d742c22288e8f5637d6d9fce4ad5c34ce61001351f4128c42dfe487324213c`）。结果必须
按两层解释：

- 严格 concrete child 覆盖 1,404/1,821 actions，完整原样 replay 覆盖 4/65；
- 经 reference 已证明安全的 cluster、DMA-lane 和 equal-load expert 对称规范化后，
  transition 覆盖 1,457/1,821，只有 5/65 条保存路径的每个 source state 都存在
  canonical-equivalent child。这个 5/65 仍不是整条替代路径证明，因为审计没有
  传播 ID remap；candidate-sufficiency 仍只能由受限图搜索或完整替代 history 给出。

对 canonical 仍缺失的 364 个 actions 做因果拆分：254 个在 seed 物理菜单中就
不存在，76 个由三种 local profile 筛选删除，20 个缺少逻辑 template，只有 14 个
是 K32 quota/order 截断。因此 `K=16 -> 24 -> 32` 的直接覆盖只小幅增加并不是
比较器宽度不足；当前第一优先级是重做物理候选表达，不能继续盲目增大 K 或提前
训练 scorer。

进一步在 344 个“逻辑 template 已存在”的真实 miss 上恢复完整物理菜单：保存
证书动作按 `FAST` 排序时只有 226/344 位于每 template 前三，仍有 34 个位于第
9--16；把 FAST、S2PF、NO_S2PF、pathmax、admissible LB、LPT、dual estimate、
DMA-release 八种规则的各自前三取并集，也只覆盖 254/344。因此不能用“增加一个
LPT/LB profile”解释或修复缺口；需要显式保留 DMA lane release/cache 的有限
Pareto physical modes，或证明一种有界 local physical selector 能安全替代它们。

当前 K32 图对代表性
`olmoe_grid_hot8_12x_dual_a29_le2_49 @ 114 ticks` 的 120 秒 exact probe 累计
1,501 expansions、OPEN=188 后超时，结果为 `unresolved`，不是 insufficient。
checkpoint 和汇总保存在 `/tmp/olmoe_t8b2_k32_exact_v1` 与
`results/policy_search/olmoe_t8b2_k32_candidate_exact_probe_v1.json`
（汇总 SHA-256
`c5d5a8880cb3f5ec9ca8f32c5e8810a08e7c5a3304d38dede125221a97e8e4ba`）。在物理
菜单重构前不继续给这个低覆盖候选图投入长时间 exact 搜索。

为防止候选器结论退化成“只要算力均分即可”，阶段 D 必须在同一
65-case 输入上增加三个可重放的分层基线：

1. `BLOCK-LPT-FIFO`：按 `ceil(ntok/2)` 将整个 expert 贪心分配给当前
   block load 更小的 cluster，然后每个 cluster 严格按分配顺序执行。
2. `BLOCK-DP-FIFO`：先用精确两分区获得最小最大 block load，但每个
   cluster 内仍按确定性降序 FIFO 执行。
3. `BLOCK-DP-DMA-GREEDY`：保持同一最优 block 分区，只在当前未执行项中
   选择局部最早完成的合法 expert/shape/DMA binding，不用 continuation
   score 或多步 lookahead。

三者都必须产生完整 `StageAction` history 并通过显式 DMA replay；计算分区
数字本身不算 schedule。这三层分别隔离贪心分配损失、执行顺序/
DMA 损失和 continuation 评分损失。

在固定窗口下依次检查：

- PAIR 的 rank 组合；
- SPLIT 的 expert rank 与 cut；
- SINGLE/one-idle 的发射对象；
- S1/S3 shape；
- S2PF/S4PF 的 OFF、single、BOTH 选择与 DMA lane；
- 必要的有限控制状态。

先运行 candidate oracle。若 oracle 仍达不到最优，补充最小必要候选；若 oracle 达到最优但策略失败，问题归入评分/控制层。

候选模板提案数据由
`analyze_window_witness_action_templates.py`（SHA-256
`35aabcacb227041410ed4bdb18b1d97d04dd6ff1eca8ad382fd15710acd54a96`）生成。
它对 audit 记录的 direct source 重新物理 replay，必要时具体化全局
一致的 equal-load ID relabel，并把 raw buffer rank 与“窗口选定后的
非 resident 等价负载类”分开统计。当前只作 pilot 的已覆盖集为
top5 60/65、三个 top8 各 63/65；完整阶段 C 结束前不固化其输出。
当前描述性结果显示 PAIR 主要是同负载类或相邻负载类，one-idle
SINGLE 则同时需要头部负载类和尾部负载类。这只用于排序待测
template axes，不用频率删除任何 action；删除必须由后续显式 DMA
candidate oracle 的因果 ablation 决定。
以当前 63 条 top8+bottom2 direct histories 为例，仅按高层 family/rank
判断，原 HW-v2 规则可表达 833/1754 个 witness actions，其余主要为
one-idle 非 rank0 SINGLE（692）、新的同步 PAIR rank（155）、standalone
PREFETCH（41）和非 rank0 SPLIT（18）。这只证明这些具体最优 histories
无法被原 bank 原样重放，不证明原 bank 不存在另一条最优路径；后者
仍由 candidate oracle 判定。
同一 top8+bottom2 集上，原 one-idle adaptive shape 阈值规则与 witness
完全一致的是 1669/2197 个已发射 cluster slots，所有已发射 slots 均一致的是
1282/1713 个 consuming actions。因此不能在 oracle 之前把 shape 固定为单一
token-threshold 规则；必须测试有限 shape profile variants 是否占用额外
candidate IDs，或能否由一个有界的本地 physical selector 等价实现。
忽略 expert/rank 后，top8+bottom2 histories 中 PAIR 有 22 种观察到的
shape/DMA/cache modes，前 4 种覆盖 80.61% PAIR actions；SINGLE 有 35 种，
两个 cluster 方向的 `C/C+BOTH` 主模式合计覆盖 77.14% SINGLE actions。
standalone PREFETCH 则同时出现 BOTH/IDMA/XDMA 三种 binding。这些仍是
描述性数据：正式实现必须将 expert/rank template 与有限 physical
profile selector 分层计数，并对 S4PF `OFF/single/BOTH` 分别做因果
ablation，不能用累积频率删除低频模式。

第一版 candidate-oracle 实验网格固定为下列有界 axes，它们是待测集合而
不是最终 bank：

- `SYNC/PAIR`：每个有至少两个可见 expert 的负载类都测试
  `(Li,Li)`；异类组合测试相邻类 `(0,1)...(4,5)`、热点 anchor
  `(0,2)/(0,3)/(0,4)`、一个 bottom-class pair，以及 concrete resident
  与有限头部类的组合。`PAIR(5,5)` 正在处理的两个 top8 未决 case 中
  都对应 `L2,L2`，因此同类 PAIR 不能被原 `(0,1)/(1,2)/(2,3)` 轴替代；
- `SYNC/SPLIT`：可见负载类 rank0--3，每类比较 HALF 与有限边缘 cut；
- `ONE-IDLE/SINGLE`：按窗口选择后的去重负载类 `L0...L6` 与具体
  resident 生成，不按所有等负载物理 entries 展开；同时给极少量
  `WAIT-PAIR` 保留独立、显式计入 K 的候选槽，只有后续因果 ablation
  证明可删才删除；
- `PREFETCH`：目标负载类与 `OFF/IDMA/XDMA/BOTH` 正交测试；
- `K=16/24/32` 在 `SYNC/ONE-IDLE/TERMINAL` 三种互斥决策状态中条件
  复用，不把不会同时合法的 families 同时计入一轮 K。

实现 oracle 时每个 axis 必须映射到确定的 `StageAction` physical profile
或一个明确有界的 local physical selector；不允许先枚举全部 reference actions
再暗中取最优而不计入 K。
当前 direct histories 中，top8 的 one-idle 决策点最多出现 7 个非 resident
去重负载类与 2 个 concrete residents；因此“每负载类两个 physical
profiles + 每 resident 一个 cached profile”的最坏候选数恰为
`7*2+2=16`。这是基于当前证书状态的上界和 K16 起点，不是第二个
physical profile 必要性证明。

standalone PREFETCH 在当前 top8+bottom2 direct histories 中出现于 12/63 个
case、共 41 个 action；目标类覆盖 `L0...L5`，binding 为 BOTH/IDMA/XDMA
各 16/15/10 次。它主要但不只出现在 SYNC 状态。因此 K16 不能同时把
`9 PAIR + 4 SPLIT + 3 PREFETCH` 当成完整候选集：PREFETCH 必须按
“目标类 x 物理 profile”真实占用 K，并与 consuming candidates 做条件配额
消融。频次只决定第一轮测试顺序，不证明任何目标类或 binding 可以删除。

正式 candidate oracle 必须直接在 `FourStageSnap`/`StageAction` 显式 DMA
reference 图上运行，不能把 reference state 投影到 `CSnap` 后调用 mirror
generator。原因是 `CSnap` 不保存 `dma_s1/dma_s3/s2pf_dma/pf_dma` lane binding，
并以 `pf_eid=-2`、`s4pf_valid` 表示尚未绑定具体 expert 的 ghost S4PF；该投影
不是无损状态同态。实现方式是在完整 reference generator 上按冻结窗口与候选
规范过滤合法 StageAction，同时保留受窗口约束的显式 PREFETCH action。窗口
尚未冻结前不提前写死 rank/cut/prefetch filter。

已确认的 median 因果样本（不提前冻结最终候选）：

1. `top4+bottom2` 已有 120-tick 最优 witness，所以该 case 的 18-tick
   top4+bottom2 策略差距不是窗口不足。
2. 根轮最优高层动作 `PAIR(22,12)` 已存在于当前 5 个 mirror 候选中；HW-v2
   和 top4+bottom2 都选择 `PAIR(6,6)`，因此第一处分歧属于评分/控制。
3. 强制正确根轮后，按最优 witness 的 expert-count/cluster 序列在现有 mirror
   候选图上逐层保留全部匹配状态；第 4 个后续动作需要向空闲 C3 发射当前
   bottom 的 1-token expert，但 17 个可达状态均只生成 6-token head。
   源码原因是现有 one-idle family 只枚举 `remaining[0]`。因此修正首轮评分后
   仍存在独立候选缺口；正式阶段 D 必须评估 one-idle 对冻结窗口内多个有限
   rank（至少 head 与 bottom）的候选扩展，而不能只修改 scorer。
4. 不改源码的受控实验只把现有 one-idle shape/start 逻辑应用到
   `top4+bottom2` 最多 6 个可见位置，其余候选类型不变。沿 32-step 最优
   expert-count/cluster 序列保留全部匹配状态后完整贯通，得到 420 个终态，
   最佳 mirror makespan=120；这说明 median 缺口可由有界 rank 枚举修复，
   不需要新增多步仿真模块。该结果仍是 mirror candidate-graph 证据，正式
   物理充分性必须由显式 DMA candidate oracle 和 replay 确认。

完成条件：对每个正式窗口形成 `K=16/24/32` 的同输入 candidate-sufficiency
矩阵和复杂度报告；冻结能达到目标覆盖的最小 K 与最小候选 family 集，而不是
先验固定某 32 个 action 模板。

#### 2026-07-29 top8+bottom2 候选器阶段性记录（已由最终 top5 方案取代）

在 top5+bottom1 最后一项仍未决期间，已完成 top8+bottom2 的独立候选充分性
与闭环删减，结论暂记为“实现候选”，不得提前写成最终窗口：

- `results/policy_search/olmoe_t8b2_k16_used_optimal_path_token_bank_v1.json`
  固化 29 个 state-relative token ROM entries，SHA-256
  `51ce72d65856609476a105da2da93387476846247ba25c7865eebea9126e1ae2`。
  它对应的显式 DMA 受限图证书为
  `olmoe_t8b2_k16_used_optimal_path_candidate_certificate_v1.json`：65/65
  candidate-sufficient，其中 6 条直接覆盖已保存最优 history，59 条找到另一条
  等周期最优路径；每轮 concrete candidate 最大为 13。这个 29-entry bank 是
  独立的候选充分性基线，后续 scorer 不得改变这项结论。
- 固定 scorer 和有界 local start selector 后，对 29 entries 做闭环
  leave-one-out 删除，得到 15-entry 实现候选
  `results/policy_search/olmoe_t8b2_bounded15_token_bank_v2.json`，SHA-256
  `76ed8b693b51dd941e2ac298463eab120038f88afd699abf206d1f44e023e7ff`。
  保留的来源 index 为
  `0,1,2,3,5,11,12,14,15,16,18,20,23,27,28`；按状态互斥复用后为
  `SYNC=6`、`ONE_IDLE=7`、`TERMINAL=2`，family 为 `PAIR=6`、`SINGLE=9`。
  没有 SPLIT、standalone PREFETCH、S4PF、WAIT-PAIR 或 SIM1。
- 15-entry 闭环不是从 action 使用频率直接截断。完整证书逐 entry 删除一次；
  15 个 entry 中没有任何一个可在保持当前 65/65 闭环最优的条件下单独删除。
  这只证明它在冻结的 65 条上是 inclusion-minimal，不宣称不存在另一组更小 ROM。
- `bounded_release` 不展开任意 future start time。PAIR 只比较两个
  expert-to-cluster placement；SINGLE/TERMINAL 只检查
  `cluster_release`、`peer_s2pf_end`、`peer_task_end` 三个事件点并丢弃非法项，
  然后以 `(max_end, sum_end, latest_start)` 选定本 token 的一个物理 child。
  静态 local variant 上限为 `SYNC=12`、`ONE_IDLE=21`、`TERMINAL=6`；65 条
  实际 materialized action 最大为 9，进入全局 scorer 的候选最大为 6。
- start-policy 因果消融保持同一 ROM/scorer：`bounded_release` 与完整
  `earliest_finish` 都是 65/65、总 gap=0；`earliest_start` 为 61/65、总 gap=12；
  保留全部 start points 为 53/65、总 gap=51；`latest_start` 为 0/65。
  因此有界事件选择不是单纯的资源优化，而是闭环正确性的一部分。

这里仍保留 29-entry candidate certificate 作为“候选空间足够”的独立证据，
15-entry certificate 则作为“固定评分器闭环可实施”的证据；二者不能互相替代。

### 阶段 E：评分器与控制策略

1. 保持窗口与候选冻结，只修改评分和 tie-break。
2. 评分输入只能来自当前 round、有限窗口、cluster/DMA/cache 状态和有限聚合量。
3. 允许从最优 histories 蒸馏整数阈值、LUT、浅层决策树或固定点线性残差，但必须闭环重新调度验证。
4. 训练/规则总结数据与最终验证数据分开报告；描述性 action 频率不能替代因果 ablation。
5. 所有 scorer 必须在相同 window、相同 K、相同 candidate IDs 上比较。报告
   65-case 的 optimal hits、累计/平均/最大 gap、逐类分布 gap、tie 数、每候选
   加法/比较/乘法或 LUT 次数、位宽与控制状态；不得用开放式 lookahead 隐藏成本。
6. 最终方案按二维目标冻结：先排除性能明显劣化方案，再在性能统计等价或接近的
   Pareto 集中选择窗口存储、候选生成、scorer datapath 和控制状态总 RTL 代价最低者。

完成条件：升级后的 Python 镜像在 65/65 条达到已证明最优周期；若存在无法在硬件约束内消除的失败，必须给出逐 case 的窗口/候选/评分证据和最小复杂度代价。

#### 2026-07-29 top8+bottom2 有界闭环历史记录（已被后续审计否决）

> `SUPERSEDED`：后续检查发现 `_practical_scalar_scorer()` 的条件顺序使
> `BOUNDED_PAIRWISE_SCORER` 命中了 full-list LPT 分支，所谓 head8-only 分支不可达。
> 因而本节的 65/65 只能证明固定 ROM 配合 full-list LPT 的表现，不能证明有界
> scorer。真正修正 dispatch 后 head8-only 为 56/65。本节保留用于记录失败来源，
> 不再作为最终实现依据。

第一轮严格评分器矩阵在固定 29-entry ROM 上比较 92 个无 lookahead 公式；最佳
纯字典序 scorer 只有 38/65、累计 gap=140 ticks。随后逐个失败边界做因果规则，
而不是把最优 target 或 teacher action 注入 runtime：one-progress 规则达到
53/65、gap=50，regime pairwise v3 达到 59/65、gap=41，最终有界版本达到
65/65、累计/最大 gap 都为 0。

最终实现候选 scorer ID 为
`lb_f_head8_compute_dma_regime_pairwise_v1`。基础 score 只由以下有界量构成：

- child pathmax `f`；
- top8 descriptor 的 two-bin LPT；
- remaining compute-work 与 DMA-capacity lower-bound；
- child 两 cluster 的完成时刻与本轮消耗 token 数；
- S2PF 数量、remaining count 和固定 candidate ID tie-break。

五个 pairwise override 只使用整数比较和固定阈值，分别处理 one-progress、
one-idle plateau、短尾 plateau、长空闲 slack-fill 和单热点/冷尾同步状态；完整
条件与优先级固化在 `OLMOE_BOUNDED_SCHEDULER_IMPLEMENTATION.md`。不存在训练模型、
乘法器权重、动态树遍历、child search、optimum target 或可变深度 lookahead。

为避免 Python 测试暗中扫描完整 remaining，runtime contract 固定维护 7 个量：
`remaining_count`、`remaining_token_sum`、`remaining_odd_count`、
`remaining_le2_count`、`remaining_shape_c_block_sum`、
`remaining_best_work_cc`、`pathmax_f_score`。对 65 条、1,935 个状态和 1,870 个
transition 的逐轮减法更新审计中，counter、完整 LB component、pathmax 与 action
trajectory mismatch 均为 0；rank selector 只从 top8+bottom2 窗口构造。

正式闭环证书为
`results/policy_search/olmoe_t8b2_bounded15_certificate_v2.json`，SHA-256
`e143838266cf402bd769ca09707203de31637e54bd8491aad73b54c703ddf6e5`；它固定当前
评估脚本 SHA-256
`3fb915e2f48027b7b48ae630db160ac6552682a61f7a6ca967add47fa6e56f04`，并证明
15-entry ROM 在 65/65 条上 terminal、optimal、总 gap=0、最大 gap=0。

干净的逐轮/整批 Python 镜像为 `scheduler_olmoe_bounded_policy.py`，策略 ID
`olmoe-t8b2-fixed15-bounded-release-pairwise-v2`；接口和 RTL lowering 规范见
`OLMOE_BOUNDED_SCHEDULER_IMPLEMENTATION.md`。两者当前仍标为 implementation
candidate 而不是 final policy，唯一原因是 top5+bottom1 的最后一个窗口
feasibility case 尚未闭合；不是因为 top8 scorer 或候选闭环仍有已知失败。

## 最终交付

1. 65 条全局最优证书与可重放 histories。
2. 现有两种调度器的同输入基线报告。
3. 窗口、候选、评分器的逐层因果 ablation。
4. 冻结的升级版 Python 镜像、策略规范和回归命令。
5. 面向后续 RTL agent 的有限状态、候选生成、评分公式与接口文档。

### 2026-07-29 最终冻结：top5+bottom1 + hist4 + fixed14

本节覆盖前文所有“top5 尚未闭合”和“top8 bounded 已冻结”的过时状态。

1. `top5+bottom1` 已通过 65/65 窗口覆盖审计。最后一条
   `olmoe_observed_ranked_window_001` 通过两个等负载 expert ID 交换得到合法
   top5 history，显式 DMA replay 为 129 ticks，与全局 `LB=UB=129` 一致。
2. top8 15-entry ROM 做 `B1 -> B0` 投影后有一个重复项，得到 14-entry ROM。
   在 top5+bottom1 上每轮实际候选最大 6。
3. head5-only 聚合 LPT 修正 dispatch 后只有 57/65。新增四个剩余 block-count
   histogram bins；冻结集合中第六热 expert 最大为 7 tokens，所以所有不可见 tail
   jobs 都在 1--4 blocks 内，head5+hist4 与完整 descending LPT 精确等价。
4. 联合审计覆盖 65 条、1,914 个状态、5,887 个候选 child、1,849 次 transition：
   LPT、lower-bound components、pathmax、counter decrement 和完整 action trajectory
   全部 0 mismatch。
5. 最终闭环为 65/65 terminal、65/65 global-optimal、累计/最大 gap=0；14 个 ROM
   entries 逐项删除均不可移除，因此对冻结集合和当前 scorer inclusion-minimal。
6. 最终 policy ID：
   `olmoe-t5b1-hist4-fixed14-bounded-release-pairwise-v1`。权威 ROM、证书和规范分别为
   `olmoe_t5b1_hist4_bounded14_token_bank_v1.json`、
   `olmoe_t5b1_hist4_bounded14_certificate_v1.json` 和
   `OLMOE_BOUNDED_SCHEDULER_IMPLEMENTATION.md`。

该冻结只完成 Python 镜像目标；RTL 尚未修改，综合面积、时序与逐轮 RTL/Python
等价仍属于后续任务。
