# HeMAiA MoE Performance Model v20 — 全策略缓存感知调度 完整分析报告

## 1. 系统配置

| 参数 | 值 |
|------|---|
| Hidden Size | 2048 |
| Shared Intermediate | 2816 |
| Routed Intermediate | 1408 |
| Routed Experts | 64 |
| TopK | 2 |
| Weight Type | INT4 |
| C0驻留 | up+half_down = 4.125MB |
| C1驻留 | gate+half_down = 4.125MB |
| 单Routed Expert | 4.125MB |

| Cluster | MAC | VC | TCDM | 用途 |
|---------|-----|----|----|------|
| C0 | 512 | 1×512 | 5MB | shared up+half_down |
| C1 | 512 | 1×512 | 5MB | shared gate+half_down |
| C2 | 512 | 2×256 | 5MB | routed expert |
| C3 | 512 | 2×256 | 5MB | routed expert + router |

SRAM xDMA: 64B/cc, iDMA: 64B/cc, P2P: 64B/cc

## 2. 静态调度策略原理

### 2.1 调度目标

- **主目标**: routed expert总执行时间 ≈ shared expert执行时间
- **次目标**: 所有cluster的VersaCore利用率高, SRAM xDMA+iDMA利用率高

### 2.2 Dual-VC硬件模型 (C2/C3) — v16修正

C2和C3各有2个256MAC VersaCore, 工作模式如下:

**Gate+Up阶段**: VC0=gate_proj, VC1=up_proj, **并行计算**
  - 每个VC做完整GEMM(M, K=2048, N=1408)
  - **每个VC独立读A和B** (无broadcast), bank需求 = 2×A_banks + 2×B_banks
  - 双VC总DMA带宽需求 = 2 × T × C × wpe bytes/cycle
  - 计算时间 = 单个GEMM时间 (因为并行)

**Down阶段**: N-split, VC0=[M,N,K/2], VC1=[M,N,K/2]
  - 每个VC独立读A和B切片, bank需求同gate+up
  - 两个VC输出concat → [M, K=2048]

**Bank模型 (v16)**: `2×A_banks + 2×B_banks`
  - 每个VC独立占用A端口和B端口, 因此总bank需求是两个VC之和
  - 示例 [4×8×8]: 2×(4+4) = 16 banks + DMA端口
  - 示例 [2×8×16]: 2×(2+8) = 20 banks + DMA端口
  - 示例 [1×8×32]: 2×(1+16) = 34 banks + DMA端口

**Per-Tile流式模型 (v16)**:
  - GEMM拆分为Mt×Nt output tiles × Kt K-tiles
  - tile0有DMA延迟暴露, 后续tile以pipeline rate处理
  - pipeline rate = max(dma_per_tile, compute×bank_stretch)
  - bank_stretch = (2×A_banks + 2×B_banks + dma_ports) / 64

### 2.3 带宽约束 (W4A8: 权重INT4, 激活INT8)

| Shape [R×T×C] | 单VC B需求 | 双VC B需求 | 2×A+2×B banks | @64B/cc | @128B/cc |
|------|------|------|------|------|------|
| [1x8x32] | 128B/cc | 256B/cc | 34 | 不足 | 不足 |
| [2x8x16] | 64B/cc | 128B/cc | 20 | 不足 | OK |
| [4x8x8] | 32B/cc | 64B/cc | 16 | OK | OK |
| [8x8x4] | 16B/cc | 32B/cc | 20 | OK | OK |
| [16x8x2] | 8B/cc | 16B/cc | 34 | OK | OK |
| [32x8x1] | 4B/cc | 8B/cc | 64 | OK | OK |

> **关键**: [4×8×8]是并行流式(64B/cc)时的最佳shape — 双VC B需求恰好=64B/cc

### 2.4 Phase-Based调度 (核心策略)

1. **Phase 1 (并行流式)**: 选一对expert, C2用sram_xDMA(64B/cc), C3用iDMA(64B/cc)并行
   - 双VC用[4×8×8], B需求=64B/cc, 恰好匹配单通道DMA带宽
   - 可拆分热门expert: 前半段流式, 后半段驻留计算
2. **Phase 2 (驻留+全BW)**: 热门expert权重已驻留, 无需DMA
   - 空闲cluster独享128B/cc, 可用[2×8×16] (双VC B需求=128B/cc)
3. **Phase 3 (清理)**: 处理剩余冷门expert (1-2 token)

## 3. 动态调度器原理

动态调度器基于cost函数, 在runtime根据TopK结果选择最优调度方案.

### 3.1 Cost函数

```
cost = |routed_cc/shared_cc - 1.0|  // 主: 时间接近
     + (1 - avg_vc_util) × 0.2     // 次: VC利用率
     + (1 - avg_sram_util) × 0.1   // 次: SRAM带宽
```

### 3.2 策略搜索池 (9种)

| # | 策略 | 核心思想 | 最适场景 |
|---|------|---------|---------|
| 1 | phase_based | 热冷配对 + expert拆分 + 驻留phase | M=1~2, 2-4 experts |
| 2 | greedy_balanced | 贪心负载均衡 @64B/cc | 多expert均匀分布 |
| 3 | sequential_full | 串行全带宽 @128B/cc | 少expert大token |
| 4 | bw_steal | 带宽窃取 — 先完成者抢DMA | 一热一冷 |
| 5 | adaptive_split | 穷举拆分点最优化 | 热门expert可拆分 |
| 6 | online_greedy | 在线贪心 — 逐步评估 | 动态DMA分配 |
| 7 | cold_batch | 冷门批量 @128B消化 | 大量1-2tok experts |
| 8 | unified_dynamic | DMA预取+专家克隆+shape切换 | 1-2 experts极多token |
| 9 | event_driven | DMA/计算解耦追踪 | M≥8通用最优 |

### 3.3 v17~v20创新机制

**v17: DMA预取 (Prefetch)**:
- 当hot expert在cluster_A计算时, DMA通道空闲(compute > DMA)
- 利用空闲DMA为cluster_B预取下一个cold expert的权重
- M=8@[4×8×8]: DMA slack=68,202cc → 可预取4.163MB (恰好一整个expert!)
- M≥16: slack更大, 可以连续预取多个expert

**v17: 专家克隆 (Expert Clone)**:
- 当只有1-2个active expert且token极多时, C2+C3各加载同一份权重
- 每个cluster处理一半的token, 实现2×计算加速
- 条件: compute_time >> 2×dma_time (加载两份权重仍比单cluster快)

**v18: 事件驱动调度器 (Event-Driven)**:
- DMA/计算解耦: DMA传输和VersaCore计算不再绑定为原子操作
- 全局时间线追踪: 维护C2/C3各自的DMA结束时刻和计算结束时刻
- 每步决策: 选择最早空闲的(cluster, DMA)对, 分配下一个expert
- M≥8时占优: 7-11% avg ratio改善

**v19: 跨层Expert缓存**:
- C2/C3各可驻留1个完整expert (4.125MB / 5MB TCDM)
- 上一MoE层的exit_eids → 下一层的cached_map
- 缓存命中时跳过DMA, 直接resident模式执行

**v20: 全策略缓存感知 + 自适应旁路**:
- 所有9种策略统一支持cached_map参数
- _preprocess_cached()将缓存命中expert分离, 以resident模式执行
- 自适应缓存旁路: 当resident节省<5%时(ntok≥4), 跳过缓存避免负载不均衡

### 3.4 缓存收益分析

| ntok | Streaming@64B/cc | Resident | 节省 | 节省比 |
|------|-----------------|----------|------|--------|
| 1 | 67,607cc | 16,993cc | 50,614cc | 74.9% |
| 2 | 67,618cc | 33,976cc | 33,642cc | 49.8% |
| 4 | 67,942cc | 67,942cc | 0cc | 0.0% |
| 8 | 135,874cc | 135,874cc | 0cc | 0.0% |

**关键结论**: ntok=1/2有显著收益(50-75%), ntok≥4几乎无收益(compute-bound)

### 3.5 自适应缓存旁路

```python
cache_benefit_ratio = cc_resident / cc_streaming
if cache_benefit_ratio > 0.95:
    # 旁路: 视为未命中, 交由策略自由调度
    uncached.append((eid, ntok))
```

### 3.6 schedule()入口

```python
def schedule(M, token_dist, sys, moe, shared_cc, cached_map=None):
    for fn in all_9_strategies:
        plan = fn(experts, sys, moe, cached_map=cached_map)
        cost = cost_function(plan, shared_cc)
        if cost < best_cost: best_plan = plan
    # 计算exit_eids供下一层使用
    for t in best_plan.tasks:
        if t.cid in (2,3): best_plan.exit_eids[t.cid] = t.eid
    return best_plan
```

## 4. 训练结果汇总

### 4.1 全局汇总表

| M | shared_cc | #Dist | NC avg | NC min | NC max | C avg | C min | ≤1.1(NC) | ≤1.1(C) | ≤1.2(NC) | ≤1.2(C) | 缓存改善 | 最优策略(C) |
|---|-----------|-------|--------|--------|--------|-------|-------|----------|---------|----------|---------|----------|-------------|
| 1 | 17,184 | 2 | 2.956 | 1.977 | 3.936 | 1.483 | 0.989 | 0% | 50% | 0% | 50% | -49.8% | phase_based |
| 2 | 34,270 | 5 | 2.572 | 1.982 | 3.947 | 1.684 | 0.991 | 0% | 20% | 0% | 20% | -34.5% | phase_based |
| 4 | 68,486 | 17 | 2.096 | 0.992 | 3.950 | 1.588 | 0.744 | 12% | 24% | 12% | 24% | -24.2% | sequential_full |
| 8 | 136,918 | 43 | 2.020 | 0.992 | 3.952 | 1.719 | 0.992 | 7% | 9% | 7% | 19% | -14.9% | event_driven |
| 16 | 273,782 | 200 | 1.719 | 0.993 | 3.952 | 1.585 | 0.806 | 5% | 14% | 20% | 36% | -7.8% | event_driven |
| 64 | 1,094,966 | 200 | 1.410 | 0.993 | 2.287 | 1.377 | 0.993 | 20% | 26% | 40% | 45% | -2.3% | event_driven |
| 128 | 2,189,878 | 100 | 1.174 | 0.993 | 1.582 | 1.163 | 0.993 | 48% | 48% | 69% | 71% | -0.9% | event_driven |

### 4.2.1 M=1 (2种分布)

- Shared CC: 17,184
- 无缓存 ratio 范围: [1.977, 3.936]
- 无缓存 avg ratio: 2.956
- 缓存100% avg ratio: 1.483
- 缓存改善: -49.8%
- ratio ≤ 1.1的比例: NC=0.0% | C=50.0%
- ratio ≤ 1.2的比例: NC=0.0% | C=50.0%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 1 | 3.936 | 3.936 | 3.936 | 0.989 | 0.989 | 0.989 | 100.0% |
| single | 1 | 1.977 | 1.977 | 1.977 | 1.977 | 1.977 | 1.977 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 2.956 | 1.483 | -49.8% ★ |
| greedy_balanced | 3.935 | 1.483 | -62.3% |
| sequential_full | 2.956 | 1.483 | -49.8% |
| bw_steal | 2.956 | 1.483 | -49.8% |
| adaptive_split | 2.956 | 1.483 | -49.8% |
| online_greedy | 2.956 | 1.483 | -49.8% |
| cold_batch | 2.956 | 1.483 | -49.8% |
| unified_dynamic | 2.956 | 1.483 | -49.8% |
| event_driven | 2.956 | 1.483 | -49.8% |

**策略胜出 (无缓存)**: phase_based:1, sequential_full:1

**策略胜出 (有缓存)**: phase_based:2

**最优案例**: ratio=0.989, dist=2experts: [1, 1], strategy=phase_based
**最差案例**: ratio=1.977, dist=1experts: [2], strategy=phase_based

### 4.2.2 M=2 (5种分布)

- Shared CC: 34,270
- 无缓存 ratio 范围: [1.982, 3.947]
- 无缓存 avg ratio: 2.572
- 缓存100% avg ratio: 1.684
- 缓存改善: -34.5%
- ratio ≤ 1.1的比例: NC=0.0% | C=20.0%
- ratio ≤ 1.2的比例: NC=0.0% | C=20.0%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 2 | 2.965 | 1.983 | 3.947 | 1.730 | 0.991 | 2.469 | 83.4% |
| hot_dominated | 2 | 2.474 | 1.982 | 2.965 | 1.487 | 1.487 | 1.487 | 100.0% |
| single | 1 | 1.983 | 1.983 | 1.983 | 1.983 | 1.983 | 1.983 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 2.569 | 1.684 | -34.5% ★ |
| greedy_balanced | 2.766 | 1.880 | -32.0% |
| sequential_full | 2.769 | 1.684 | -39.2% |
| bw_steal | 2.766 | 1.683 | -39.1% |
| adaptive_split | 2.569 | 1.683 | -34.5% |
| online_greedy | 2.570 | 1.683 | -34.5% |
| cold_batch | 2.769 | 1.684 | -39.2% |
| unified_dynamic | 2.570 | 1.683 | -34.5% |
| event_driven | 2.569 | 1.683 | -34.5% |

**策略胜出 (无缓存)**: sequential_full:3, phase_based:2

**策略胜出 (有缓存)**: phase_based:3, sequential_full:2

**最优案例**: ratio=0.991, dist=2experts: [2, 2], strategy=phase_based
**最差案例**: ratio=2.469, dist=4experts: [1, 1, 1, 1], strategy=sequential_full

### 4.2.3 M=4 (17种分布)

- Shared CC: 68,486
- 无缓存 ratio 范围: [0.992, 3.950]
- 无缓存 avg ratio: 2.096
- 缓存100% avg ratio: 1.588
- 缓存改善: -24.2%
- ratio ≤ 1.1的比例: NC=11.8% | C=23.5%
- ratio ≤ 1.2的比例: NC=11.8% | C=23.5%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 3 | 2.309 | 0.992 | 3.950 | 1.732 | 0.992 | 3.211 | 85.8% |
| hot_dominated | 12 | 2.021 | 1.486 | 2.967 | 1.508 | 0.744 | 2.228 | 85.4% |
| cold_dominated | 1 | 3.459 | 3.459 | 3.459 | 2.719 | 2.719 | 2.719 | 66.9% |
| single | 1 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 2.240 | 1.938 | -13.5% |
| greedy_balanced | 2.442 | 1.921 | -21.3% |
| sequential_full | 2.679 | 1.997 | -25.5% ★ |
| bw_steal | 2.413 | 1.863 | -22.8% |
| adaptive_split | 2.196 | 1.733 | -21.1% |
| online_greedy | 2.386 | 1.791 | -24.9% |
| cold_batch | 2.621 | 1.938 | -26.0% |
| unified_dynamic | 2.269 | 1.733 | -23.6% |
| event_driven | 2.108 | 1.689 | -19.9% |

**策略胜出 (无缓存)**: phase_based:7, sequential_full:5, event_driven:3

**策略胜出 (有缓存)**: sequential_full:6, adaptive_split:3, phase_based:2

**最优案例**: ratio=0.744, dist=4experts: [3, 2, 2, 1], strategy=online_greedy
**最差案例**: ratio=3.211, dist=8experts: [1, 1, 1, 1, 1, 1, 1, 1], strategy=sequential_full

### 4.2.4 M=8 (43种分布)

- Shared CC: 136,918
- 无缓存 ratio 范围: [0.992, 3.952]
- 无缓存 avg ratio: 2.020
- 缓存100% avg ratio: 1.719
- 缓存改善: -14.9%
- ratio ≤ 1.1的比例: NC=7.0% | C=9.3%
- ratio ≤ 1.2的比例: NC=7.0% | C=18.6%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 4 | 1.980 | 0.992 | 3.952 | 1.763 | 0.992 | 3.582 | 80.1% |
| hot_dominated | 19 | 1.629 | 1.239 | 2.225 | 1.376 | 0.993 | 1.979 | 79.8% |
| cold_dominated | 10 | 2.868 | 2.231 | 3.706 | 2.461 | 1.861 | 3.336 | 65.8% |
| mixed | 9 | 2.034 | 1.486 | 2.474 | 1.678 | 1.240 | 2.104 | 66.7% |
| single | 1 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 2.174 | 2.079 | -4.4% |
| greedy_balanced | 2.231 | 1.928 | -13.6% |
| sequential_full | 2.649 | 2.299 | -13.2% |
| bw_steal | 2.226 | 1.911 | -14.2% |
| adaptive_split | 2.151 | 1.848 | -14.1% |
| online_greedy | 2.261 | 1.934 | -14.5% |
| cold_batch | 2.453 | 2.114 | -13.8% |
| unified_dynamic | 2.192 | 1.865 | -14.9% |
| event_driven | 2.018 | 1.752 | -13.2% ★ |

**策略胜出 (无缓存)**: phase_based:15, event_driven:12, sequential_full:9

**策略胜出 (有缓存)**: event_driven:14, sequential_full:8, greedy_balanced:8

**最优案例**: ratio=0.992, dist=1experts: [16], strategy=unified_dynamic
**最差案例**: ratio=3.582, dist=16experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(6 more), strategy=sequential_full

### 4.2.5 M=16 (200种分布)

- Shared CC: 273,782
- 无缓存 ratio 范围: [0.993, 3.952]
- 无缓存 avg ratio: 1.719
- 缓存100% avg ratio: 1.585
- 缓存改善: -7.8%
- ratio ≤ 1.1的比例: NC=5.0% | C=14.5%
- ratio ≤ 1.2的比例: NC=20.5% | C=36.5%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 5 | 1.783 | 0.993 | 3.952 | 1.695 | 0.993 | 3.767 | 81.9% |
| hot_dominated | 53 | 1.251 | 0.993 | 2.099 | 1.164 | 0.806 | 1.914 | 90.8% |
| cold_dominated | 44 | 2.621 | 1.737 | 3.829 | 2.431 | 1.484 | 3.644 | 55.1% |
| mixed | 97 | 1.570 | 1.116 | 2.596 | 1.431 | 1.055 | 2.412 | 75.1% |
| single | 1 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 1.854 | 1.846 | -0.4% |
| greedy_balanced | 1.871 | 1.733 | -7.4% |
| sequential_full | 2.457 | 2.226 | -9.4% |
| bw_steal | 1.885 | 1.742 | -7.6% |
| adaptive_split | 1.799 | 1.682 | -6.5% |
| online_greedy | 1.902 | 1.579 | -17.0% |
| cold_batch | 2.115 | 1.958 | -7.4% |
| unified_dynamic | 1.835 | 1.702 | -7.2% |
| event_driven | 1.724 | 1.599 | -7.3% ★ |

**策略胜出 (无缓存)**: event_driven:79, phase_based:60, sequential_full:17

**策略胜出 (有缓存)**: event_driven:83, adaptive_split:34, greedy_balanced:28

**最优案例**: ratio=0.806, dist=4experts: [10, 10, 9, 3], strategy=online_greedy
**最差案例**: ratio=3.767, dist=32experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(22 more), strategy=sequential_full

### 4.2.6 M=64 (200种分布)

- Shared CC: 1,094,966
- 无缓存 ratio 范围: [0.993, 2.287]
- 无缓存 avg ratio: 1.410
- 缓存100% avg ratio: 1.377
- 缓存改善: -2.3%
- ratio ≤ 1.1的比例: NC=19.5% | C=25.5%
- ratio ≤ 1.2的比例: NC=39.5% | C=45.0%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 6 | 1.158 | 0.993 | 1.986 | 1.153 | 0.993 | 1.955 | 100.0% |
| hot_dominated | 21 | 1.313 | 1.054 | 1.979 | 1.283 | 1.008 | 1.899 | 82.1% |
| cold_dominated | 28 | 2.014 | 1.485 | 2.287 | 1.965 | 1.439 | 2.304 | 58.1% |
| mixed | 144 | 1.320 | 1.024 | 1.978 | 1.288 | 1.008 | 1.932 | 80.5% |
| single | 1 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 1.546 | 1.599 | +3.4% |
| greedy_balanced | 1.542 | 1.509 | -2.1% |
| sequential_full | 2.309 | 2.252 | -2.5% |
| bw_steal | 1.546 | 1.512 | -2.2% |
| adaptive_split | 1.512 | 1.486 | -1.7% |
| online_greedy | 1.577 | 1.403 | -11.0% |
| cold_batch | 1.798 | 1.762 | -2.0% |
| unified_dynamic | 1.517 | 1.483 | -2.2% |
| event_driven | 1.408 | 1.379 | -2.1% ★ |

**策略胜出 (无缓存)**: event_driven:105, phase_based:48, unified_dynamic:11

**策略胜出 (有缓存)**: event_driven:125, greedy_balanced:19, adaptive_split:19

**最优案例**: ratio=0.993, dist=1experts: [128], strategy=unified_dynamic
**最差案例**: ratio=2.304, dist=64experts: [11, 11, 11, 11, 10, 10, 2, 2, 2, 2]...(54 more), strategy=cold_batch

### 4.2.7 M=128 (100种分布)

- Shared CC: 2,189,878
- 无缓存 ratio 范围: [0.993, 1.582]
- 无缓存 avg ratio: 1.174
- 缓存100% avg ratio: 1.163
- 缓存改善: -0.9%
- ratio ≤ 1.1的比例: NC=48.0% | C=48.0%
- ratio ≤ 1.2的比例: NC=69.0% | C=71.0%

| 分布模式 | 样本数 | NC avg | NC best | NC worst | C avg | C best | C worst | 平均VC利用率 |
|---------|--------|--------|---------|----------|-------|--------|---------|------------|
| all_uniform | 6 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |
| hot_dominated | 14 | 1.288 | 1.024 | 1.582 | 1.268 | 1.024 | 1.586 | 81.8% |
| cold_dominated | 19 | 1.373 | 1.008 | 1.516 | 1.356 | 1.008 | 1.494 | 74.8% |
| mixed | 60 | 1.105 | 1.016 | 1.455 | 1.098 | 1.008 | 1.408 | 91.6% |
| single | 1 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |

| 策略 | NC avg | C100 avg | 改善% |
|------|--------|----------|-------|
| phase_based | 1.367 | 1.435 | +5.0% |
| greedy_balanced | 1.340 | 1.327 | -1.0% |
| sequential_full | 2.147 | 2.112 | -1.6% |
| bw_steal | 1.356 | 1.343 | -1.0% |
| adaptive_split | 1.334 | 1.321 | -1.0% |
| online_greedy | 1.393 | 1.379 | -1.0% |
| cold_batch | 1.402 | 1.388 | -1.0% |
| unified_dynamic | 1.311 | 1.299 | -1.0% |
| event_driven | 1.171 | 1.170 | -0.0% ★ |

**策略胜出 (无缓存)**: event_driven:63, phase_based:18, unified_dynamic:17

**策略胜出 (有缓存)**: event_driven:68, unified_dynamic:16, phase_based:7

**最优案例**: ratio=0.993, dist=1experts: [256], strategy=unified_dynamic
**最差案例**: ratio=1.586, dist=53experts: [204, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(43 more), strategy=adaptive_split

## 5. 静态LUT (查找表)

### 5.1 Per-Mode Per-M LUT

| M | 分布模式 | 推荐策略(NC) | 推荐策略(C) | NC avg | NC best | NC worst | C avg | C best | C worst | VC利用率(C) |
|---|---------|-------------|------------|--------|---------|----------|-------|--------|---------|------------|
| 1 | all_uniform | sequential_full | phase_based | 3.936 | 3.936 | 3.936 | 0.989 | 0.989 | 0.989 | 100.0% |
| 1 | single | phase_based | phase_based | 1.977 | 1.977 | 1.977 | 1.977 | 1.977 | 1.977 | 100.0% |
| 2 | all_uniform | sequential_full | phase_based | 2.965 | 1.983 | 3.947 | 1.730 | 0.991 | 2.469 | 83.4% |
| 2 | hot_dominated | phase_based | phase_based | 2.474 | 1.982 | 2.965 | 1.487 | 1.487 | 1.487 | 100.0% |
| 2 | single | phase_based | phase_based | 1.983 | 1.983 | 1.983 | 1.983 | 1.983 | 1.983 | 100.0% |
| 4 | all_uniform | phase_based | event_driven | 2.309 | 0.992 | 3.950 | 1.732 | 0.992 | 3.211 | 85.8% |
| 4 | hot_dominated | adaptive_split | greedy_balanced | 2.021 | 1.486 | 2.967 | 1.508 | 0.744 | 2.228 | 85.4% |
| 4 | cold_dominated | sequential_full | sequential_full | 3.459 | 3.459 | 3.459 | 2.719 | 2.719 | 2.719 | 66.9% |
| 4 | single | unified_dynamic | unified_dynamic | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 100.0% |
| 8 | all_uniform | phase_based | phase_based | 1.980 | 0.992 | 3.952 | 1.763 | 0.992 | 3.582 | 80.1% |
| 8 | hot_dominated | event_driven | event_driven | 1.629 | 1.239 | 2.225 | 1.376 | 0.993 | 1.979 | 79.8% |
| 8 | cold_dominated | sequential_full | sequential_full | 2.868 | 2.231 | 3.706 | 2.461 | 1.861 | 3.336 | 65.8% |
| 8 | mixed | phase_based | greedy_balanced | 2.034 | 1.486 | 2.474 | 1.678 | 1.240 | 2.104 | 66.7% |
| 8 | single | unified_dynamic | unified_dynamic | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 100.0% |
| 16 | all_uniform | phase_based | phase_based | 1.783 | 0.993 | 3.952 | 1.695 | 0.993 | 3.767 | 81.9% |
| 16 | hot_dominated | event_driven | unified_dynamic | 1.251 | 0.993 | 2.099 | 1.164 | 0.806 | 1.914 | 90.8% |
| 16 | cold_dominated | cold_batch | event_driven | 2.621 | 1.737 | 3.829 | 2.431 | 1.484 | 3.644 | 55.1% |
| 16 | mixed | event_driven | event_driven | 1.570 | 1.116 | 2.596 | 1.431 | 1.055 | 2.412 | 75.1% |
| 16 | single | unified_dynamic | unified_dynamic | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |
| 64 | all_uniform | phase_based | phase_based | 1.158 | 0.993 | 1.986 | 1.153 | 0.993 | 1.955 | 100.0% |
| 64 | hot_dominated | adaptive_split | unified_dynamic | 1.313 | 1.054 | 1.979 | 1.283 | 1.008 | 1.899 | 82.1% |
| 64 | cold_dominated | event_driven | event_driven | 2.014 | 1.485 | 2.287 | 1.965 | 1.439 | 2.304 | 58.1% |
| 64 | mixed | event_driven | event_driven | 1.320 | 1.024 | 1.978 | 1.288 | 1.008 | 1.932 | 80.5% |
| 64 | single | unified_dynamic | unified_dynamic | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |
| 128 | all_uniform | phase_based | phase_based | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |
| 128 | hot_dominated | unified_dynamic | unified_dynamic | 1.288 | 1.024 | 1.582 | 1.268 | 1.024 | 1.586 | 81.8% |
| 128 | cold_dominated | unified_dynamic | unified_dynamic | 1.373 | 1.008 | 1.516 | 1.356 | 1.008 | 1.494 | 74.8% |
| 128 | mixed | event_driven | event_driven | 1.105 | 1.016 | 1.455 | 1.098 | 1.008 | 1.408 | 91.6% |
| 128 | single | unified_dynamic | unified_dynamic | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 100.0% |

### 5.2 简化LUT (仅M维度)

| M | 无缓存推荐策略 | 有缓存推荐策略 |
|---|---------------|---------------|
| 1 | phase_based | phase_based |
| 2 | sequential_full | phase_based |
| 4 | phase_based | sequential_full |
| 8 | phase_based | event_driven |
| 16 | event_driven | event_driven |
| 64 | event_driven | event_driven |
| 128 | event_driven | event_driven |

## 6. 详细任务流表

---

**M=1 最优案例** (ratio=1.694)

### M=1 任务流表 (dist: 2experts: [1, 1])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 32 | 32 |  |  |  |  |  |  |  |  |  | token_A→C0 (2048B) |  |  |  |  | token_A→C1 (2048B) |  |
| 32 | 64 | 32 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (2048B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 64 | 1,056 | 992 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,056 | 2,089 | 1,033 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  | router [1,2048,64] |  |  |  |  |  |  |  |  |  |
| 2,089 | 7,089 | 5,000 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 7,089 | 11,389 | 4,300 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,389 | 11,411 | 22 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (2816 elem) |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,411 | 11,433 | 22 |  | up_result P2P C0→C1 (pipeline) |  |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,433 | 11,455 | 22 |  |  |  | C1 GLU (2816 elem) |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,455 | 11,499 | 44 |  |  | C1 half_down [1,2816,1024] |  |  |  |  |  | active_A C1→C0 (2816B) |  |  | scatter |  |  |  |  |
| 11,499 | 12,089 | 590 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 12,089 | 12,121 | 32 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (2048B) | token_A→C2 (2048B) |  |  |
| 12,121 | 17,124 | 5,003 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 17,124 | 17,168 | 44 | C0 half_down [1,2816,1024] |  |  |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 17,168 | 17,184 | 16 |  |  |  |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  | merge half_down (1024B) |  |  | softmax |  |  |  |  |
| 17,184 | 23,434 | 6,250 |  |  |  |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 23,434 | 23,445 | 11 |  |  |  |  |  | E0 SwiGLU |  | E1 SwiGLU |  |  |  | softmax |  |  |  |  |
| 23,445 | 27,089 | 3,644 |  |  |  |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  | softmax |  |  |  |  |
| 27,089 | 29,114 | 2,025 |  |  |  |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=1)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 17,184 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 17,184 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=1)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (2048B) | DMA_sram_xDMA↔C0_xDMA | 0 | 32 | 32 | ceil(2048/64)=32 |
| 1 | token_A→C1 (2048B) | iDMA→C1 | 0 | 32 | 32 | ceil(2048/64)=32 |
| 2 | C0 up_proj [1,2048,2816] | C0_VC | 32 | 11,389 | 11,357 | gemm(1,2048,2816,[1x8x64])=11357 util=100% |
| 3 | C1 gate_proj [1,2048,2816] | C1_VC | 32 | 11,389 | 11,357 | gemm(1,2048,2816,[1x8x64])=11357 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 32 | 11,433 | 11,401 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 32 | 1,056 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (2048B) | DMA_sram_xDMA↔C3_xDMA | 32 | 64 | 32 | ceil(2048/64)=32 |
| 7 | router [1,2048,64] | C3_VC | 1,056 | 2,089 | 1,033 | gemm(1,2048,64,[2x8x16])=1033 util=50% |
| 8 | topK | Host | 2,089 | 7,089 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 7,089 | 12,089 | 5,000 | ~5000cc |
| 10 | C1 SiLU (2816 elem) | C1_elem | 11,389 | 11,411 | 22 | ceil(2816/128)=22 |
| 11 | C1 GLU (2816 elem) | C1_elem | 11,433 | 11,455 | 22 | ceil(2816/128)=22 |
| 12 | active_A C1→C0 (2816B) | DMA_C1_xDMA↔C0_xDMA | 11,455 | 11,499 | 44 | ceil(2816/64)=44 |
| 13 | C1 half_down [1,2816,1024] | C1_VC | 11,455 | 17,124 | 5,669 | gemm(1,2816,1024,[1x8x64])=5669 util=100% |
| 14 | C0 half_down [1,2816,1024] | C0_VC | 11,499 | 17,168 | 5,669 | gemm(1,2816,1024,[1x8x64])=5669 util=100% |
| 15 | softmax | Host | 12,089 | 27,089 | 15,000 | ~15000cc |
| 16 | token_A→C2 (2048B) | SRAM(xDMA)→C2 | 12,089 | 12,121 | 32 | ceil(2048/64)=32 [xDMA] |
| 17 | token_A→C3 (2048B) | SRAM(iDMA)→C3 | 12,089 | 12,121 | 32 | ceil(2048/64)=32 [iDMA] |
| 18 | E0 gate+up [1,2048,1408] resident | C2_VC | 12,121 | 23,434 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 19 | E1 gate+up [1,2048,1408] resident | C3_VC | 12,121 | 23,434 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 20 | merge half_down (1024B) | DMA_C1_xDMA↔C0_xDMA | 17,168 | 17,184 | 16 | ceil(1024/64)=16 |
| 21 | E0 SwiGLU | C2_elem | 23,434 | 23,445 | 11 | ceil(1408/128)=11 |
| 22 | E1 SwiGLU | C3_elem | 23,434 | 23,445 | 11 | ceil(1408/128)=11 |
| 23 | E0 down [1,1408,1024]×2 resident | C2_VC | 23,445 | 29,114 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 24 | E1 down [1,1408,1024]×2 resident | C3_VC | 23,445 | 29,114 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |

#### 调度决策表 (M=1, 策略=phase_based)

- Token分布: 2experts: [1, 1]
- Routed CC: 29,114, Shared CC: 17,184, Ratio: 1.694
- VC利用率: 100.0%, xDMA利用率: 0.0%, iDMA利用率: 0.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E1 | 1 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C3 (省75%) |

---

**M=1 中位案例** (ratio=2.683)

### M=1 任务流表 (dist: 1experts: [2])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 32 | 32 |  |  |  |  |  |  |  |  | token_A→C0 (2048B) |  |  |  |  | token_A→C1 (2048B) |  |
| 32 | 64 | 32 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  | token_A sram→C3 (2048B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 64 | 1,056 | 992 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,056 | 2,089 | 1,033 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  | router [1,2048,64] |  |  |  |  |  |  |  |  |
| 2,089 | 7,089 | 5,000 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  | topK |  |  |  |  |
| 7,089 | 11,389 | 4,300 | C0 up_proj [1,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [1,2048,2816] |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,389 | 11,411 | 22 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (2816 elem) |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,411 | 11,433 | 22 |  | up_result P2P C0→C1 (pipeline) |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,433 | 11,455 | 22 |  |  |  | C1 GLU (2816 elem) |  |  |  |  |  |  | scatter |  |  |  |  |
| 11,455 | 11,499 | 44 |  |  | C1 half_down [1,2816,1024] |  |  |  |  | active_A C1→C0 (2816B) |  |  | scatter |  |  |  |  |
| 11,499 | 12,089 | 590 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 12,089 | 12,121 | 32 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  |  |  |  |  |  |  | softmax | token_A→C3 (2048B) | token_A→C2 (2048B) |  |  |
| 12,121 | 17,124 | 5,003 | C0 half_down [1,2816,1024] |  | C1 half_down [1,2816,1024] |  | E0 gate+up [2,2048,1408] resid |  |  |  |  |  | softmax |  |  |  |  |
| 17,124 | 17,168 | 44 | C0 half_down [1,2816,1024] |  |  |  | E0 gate+up [2,2048,1408] resid |  |  |  |  |  | softmax |  |  |  |  |
| 17,168 | 17,184 | 16 |  |  |  |  | E0 gate+up [2,2048,1408] resid |  |  | merge half_down (1024B) |  |  | softmax |  |  |  |  |
| 17,184 | 27,089 | 9,905 |  |  |  |  | E0 gate+up [2,2048,1408] resid |  |  |  |  |  | softmax |  |  |  |  |
| 27,089 | 34,742 | 7,653 |  |  |  |  | E0 gate+up [2,2048,1408] resid |  |  |  |  |  |  |  |  |  |  |
| 34,742 | 34,764 | 22 |  |  |  |  |  | E0 SwiGLU |  |  |  |  |  |  |  |  |  |
| 34,764 | 46,097 | 11,333 |  |  |  |  | E0 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=1)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 17,184 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 17,184 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=1)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (2048B) | DMA_sram_xDMA↔C0_xDMA | 0 | 32 | 32 | ceil(2048/64)=32 |
| 1 | token_A→C1 (2048B) | iDMA→C1 | 0 | 32 | 32 | ceil(2048/64)=32 |
| 2 | C0 up_proj [1,2048,2816] | C0_VC | 32 | 11,389 | 11,357 | gemm(1,2048,2816,[1x8x64])=11357 util=100% |
| 3 | C1 gate_proj [1,2048,2816] | C1_VC | 32 | 11,389 | 11,357 | gemm(1,2048,2816,[1x8x64])=11357 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 32 | 11,433 | 11,401 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 32 | 1,056 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (2048B) | DMA_sram_xDMA↔C3_xDMA | 32 | 64 | 32 | ceil(2048/64)=32 |
| 7 | router [1,2048,64] | C3_VC | 1,056 | 2,089 | 1,033 | gemm(1,2048,64,[2x8x16])=1033 util=50% |
| 8 | topK | Host | 2,089 | 7,089 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 7,089 | 12,089 | 5,000 | ~5000cc |
| 10 | C1 SiLU (2816 elem) | C1_elem | 11,389 | 11,411 | 22 | ceil(2816/128)=22 |
| 11 | C1 GLU (2816 elem) | C1_elem | 11,433 | 11,455 | 22 | ceil(2816/128)=22 |
| 12 | active_A C1→C0 (2816B) | DMA_C1_xDMA↔C0_xDMA | 11,455 | 11,499 | 44 | ceil(2816/64)=44 |
| 13 | C1 half_down [1,2816,1024] | C1_VC | 11,455 | 17,124 | 5,669 | gemm(1,2816,1024,[1x8x64])=5669 util=100% |
| 14 | C0 half_down [1,2816,1024] | C0_VC | 11,499 | 17,168 | 5,669 | gemm(1,2816,1024,[1x8x64])=5669 util=100% |
| 15 | softmax | Host | 12,089 | 27,089 | 15,000 | ~15000cc |
| 16 | token_A→C2 (2048B) | SRAM(xDMA)→C2 | 12,089 | 12,121 | 32 | ceil(2048/64)=32 [xDMA] |
| 17 | token_A→C3 (2048B) | SRAM(iDMA)→C3 | 12,089 | 12,121 | 32 | ceil(2048/64)=32 [iDMA] |
| 18 | E0 gate+up [2,2048,1408] resident | C2_VC | 12,121 | 34,742 | 22,621 | dual_vc_gu_resident: gemm(2,2048,1408,[1x8x32])=22621 util=100% |
| 19 | merge half_down (1024B) | DMA_C1_xDMA↔C0_xDMA | 17,168 | 17,184 | 16 | ceil(1024/64)=16 |
| 20 | E0 SwiGLU | C2_elem | 34,742 | 34,764 | 22 | ceil(2816/128)=22 |
| 21 | E0 down [2,1408,1024]×2 resident | C2_VC | 34,764 | 46,097 | 11,333 | dual_vc_dn_resident: gemm(2,1408,1024,[1x8x32])=11333 util=100% |

#### 调度决策表 (M=1, 策略=phase_based)

- Token分布: 1experts: [2]
- Routed CC: 46,097, Shared CC: 17,184, Ratio: 2.683
- VC利用率: 100.0%, xDMA利用率: 0.0%, iDMA利用率: 0.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 2 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 33,976 | 缓存命中 resident 2tok @C2 (省50%) |

---

**M=4 最优案例** (ratio=0.939)

### M=4 任务流表 (dist: 4experts: [3, 2, 2, 1])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 128 |  |  |  |  |  |  |  |  |  | token_A→C0 (8192B) |  |  |  |  | token_A→C1 (8192B) |  |
| 128 | 256 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (8192B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 256 | 1,152 | 896 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,152 | 3,213 | 2,061 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  | router [4,2048,64] |  |  |  |  |  |  |  |  |  |
| 3,213 | 8,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 8,213 | 13,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 13,213 | 13,341 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (8192B) | token_A→C2 (8192B) |  |  |
| 13,341 | 24,654 | 11,313 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E3 gate+up [1,2048,1408] resid |  | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 24,654 | 24,665 | 11 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  | E3 SwiGLU | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 24,665 | 28,213 | 3,548 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E3 down [1,1408,1024]×2 reside |  | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 28,213 | 30,334 | 2,121 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E3 down [1,1408,1024]×2 reside |  | E1 gate+up [2,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 30,334 | 35,962 | 5,628 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E2 gate+up [2,2048,1408] strea |  | E1 gate+up [2,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 35,962 | 35,984 | 22 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E2 gate+up [2,2048,1408] strea |  |  | E1 SwiGLU |  |  |  |  |  |  |  |  |
| 35,984 | 45,541 | 9,557 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 45,541 | 45,585 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (11264 elem) | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 45,585 | 45,629 | 44 |  |  |  | C1 SiLU (11264 elem) | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 45,629 | 45,717 | 88 |  |  |  | C1 GLU (11264 elem) | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 45,717 | 45,761 | 44 |  |  | C1 half_down [4,2816,1024] |  | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,761 | 45,893 | 132 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,893 | 47,317 | 1,424 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 gate+up [2,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 47,317 | 52,955 | 5,638 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 52,955 | 52,977 | 22 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 52,977 | 64,310 | 11,333 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 64,310 | 68,378 | 4,068 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68,378 | 68,422 | 44 | C0 half_down [4,2816,1024] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68,422 | 68,486 | 64 |  |  |  |  |  |  |  |  | merge half_down (4096B) |  |  |  |  |  |  |  |

#### TCDM状态 (M=4)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 68,486 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 68,486 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 64,310 | C2 | E2_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=4)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (8192B) | DMA_sram_xDMA↔C0_xDMA | 0 | 128 | 128 | ceil(8192/64)=128 |
| 1 | token_A→C1 (8192B) | iDMA→C1 | 0 | 128 | 128 | ceil(8192/64)=128 |
| 2 | C0 up_proj [4,2048,2816] | C0_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 3 | C1 gate_proj [4,2048,2816] | C1_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 128 | 45,585 | 45,457 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 128 | 1,152 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (8192B) | DMA_sram_xDMA↔C3_xDMA | 128 | 256 | 128 | ceil(8192/64)=128 |
| 7 | router [4,2048,64] | C3_VC | 1,152 | 3,213 | 2,061 | gemm(4,2048,64,[2x8x16])=2061 util=100% |
| 8 | topK | Host | 3,213 | 8,213 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 8,213 | 13,213 | 5,000 | ~5000cc |
| 10 | softmax | Host | 13,213 | 28,213 | 15,000 | ~15000cc |
| 11 | token_A→C2 (8192B) | SRAM(xDMA)→C2 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [xDMA] |
| 12 | token_A→C3 (8192B) | SRAM(iDMA)→C3 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [iDMA] |
| 13 | E1 gate+up [2,2048,1408] resident | C3_VC | 13,341 | 35,962 | 22,621 | dual_vc_gu_resident: gemm(2,2048,1408,[1x8x32])=22621 util=100% |
| 14 | E3 gate+up [1,2048,1408] resident | C2_VC | 13,341 | 24,654 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 15 | E3 SwiGLU | C2_elem | 24,654 | 24,665 | 11 | ceil(1408/128)=11 |
| 16 | E3 down [1,1408,1024]×2 resident | C2_VC | 24,665 | 30,334 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 17 | E2 gate+up [2,2048,1408] stream (dual-VC | C2_VC | 30,334 | 52,955 | 22,621 | pertile: 88tiles tile0=257 pipe=257 dma_total=22528 bank_s=1.00 →22621 [compute-bound] |
| 18 | E1 SwiGLU | C3_elem | 35,962 | 35,984 | 22 | ceil(2816/128)=22 |
| 19 | E1 down [2,1408,1024]×2 resident | C3_VC | 35,984 | 47,317 | 11,333 | dual_vc_dn_resident: gemm(2,1408,1024,[1x8x32])=11333 util=100% |
| 20 | C1 SiLU (11264 elem) | C1_elem | 45,541 | 45,629 | 88 | ceil(11264/128)=88 |
| 21 | C1 GLU (11264 elem) | C1_elem | 45,629 | 45,717 | 88 | ceil(11264/128)=88 |
| 22 | active_A C1→C0 (11264B) | DMA_C1_xDMA↔C0_xDMA | 45,717 | 45,893 | 176 | ceil(11264/64)=176 |
| 23 | C1 half_down [4,2816,1024] | C1_VC | 45,717 | 68,378 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 24 | C0 half_down [4,2816,1024] | C0_VC | 45,761 | 68,422 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 25 | E2 SwiGLU | C2_elem | 52,955 | 52,977 | 22 | ceil(2816/128)=22 |
| 26 | E2 down [2,1408,1024]×2 stream (dual-VC  | C2_VC | 52,977 | 64,310 | 11,333 | pertile: 64tiles tile0=177 pipe=177 dma_total=11264 bank_s=1.00 →11333 [compute-bound] |
| 27 | merge half_down (4096B) | DMA_C1_xDMA↔C0_xDMA | 68,422 | 68,486 | 64 | ceil(4096/64)=64 |

#### 调度决策表 (M=4, 策略=online_greedy)

- Token分布: 4experts: [3, 2, 2, 1]
- Routed CC: 64,310, Shared CC: 68,486, Ratio: 0.939
- VC利用率: 100.0%, xDMA利用率: 66.7%, iDMA利用率: 66.7%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E1 | 2 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 33,976 | 缓存命中 resident 2tok @C3 (省50%) |
| E3 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E2 | 2 | C2 | [2x8x16] | both | 128 | 1 | 否 | 100% | 33,976 | online_greedy @128B/cc |

---

**M=4 中位案例** (ratio=2.093)

### M=4 任务流表 (dist: 3experts: [6, 1, 1])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 128 |  |  |  |  |  |  |  |  |  | token_A→C0 (8192B) |  |  |  |  | token_A→C1 (8192B) |  |
| 128 | 256 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (8192B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 256 | 1,152 | 896 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,152 | 3,213 | 2,061 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  | router [4,2048,64] |  |  |  |  |  |  |  |  |  |
| 3,213 | 8,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 8,213 | 13,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 13,213 | 13,341 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (8192B) | token_A→C2 (8192B) |  |  |
| 13,341 | 28,213 | 14,872 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  | softmax |  |  |  |  |
| 28,213 | 45,541 | 17,328 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,541 | 45,585 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (11264 elem) | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,585 | 45,629 | 44 |  |  |  | C1 SiLU (11264 elem) | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,629 | 45,717 | 88 |  |  |  | C1 GLU (11264 elem) | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,717 | 45,761 | 44 |  |  | C1 half_down [4,2816,1024] |  | E0 gate+up [6,2048,1408] strea |  |  |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,761 | 45,893 | 132 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E0 gate+up [6,2048,1408] strea |  |  |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,893 | 68,378 | 22,485 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 68,378 | 68,422 | 44 | C0 half_down [4,2816,1024] |  |  |  | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 68,422 | 68,486 | 64 |  |  |  |  | E0 gate+up [6,2048,1408] strea |  |  |  | merge half_down (4096B) |  |  |  |  |  |  |  |
| 68,486 | 81,194 | 12,708 |  |  |  |  | E0 gate+up [6,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 81,194 | 81,260 | 66 |  |  |  |  |  | E0 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 81,260 | 92,524 | 11,264 |  |  |  |  | E0 down [6,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 92,524 | 115,058 | 22,534 |  |  |  |  | E0 down [6,1408,1024]×2 stream |  | E1 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 115,058 | 115,069 | 11 |  |  |  |  | E0 down [6,1408,1024]×2 stream |  |  | E1 SwiGLU |  |  |  |  |  |  |  |  |
| 115,069 | 115,249 | 180 |  |  |  |  | E0 down [6,1408,1024]×2 stream |  | E1 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 115,249 | 126,339 | 11,090 |  |  |  |  |  |  | E1 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 126,339 | 137,652 | 11,313 |  |  |  |  |  |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 137,652 | 137,663 | 11 |  |  |  |  |  |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |
| 137,663 | 143,332 | 5,669 |  |  |  |  |  |  | E2 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=4)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 68,486 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 68,486 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 115,249 | C2 | E0_weights:4.125MB | 4.125MB | 0.875MB |
| 126,339 | C3 | E1_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=4)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (8192B) | DMA_sram_xDMA↔C0_xDMA | 0 | 128 | 128 | ceil(8192/64)=128 |
| 1 | token_A→C1 (8192B) | iDMA→C1 | 0 | 128 | 128 | ceil(8192/64)=128 |
| 2 | C0 up_proj [4,2048,2816] | C0_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 3 | C1 gate_proj [4,2048,2816] | C1_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 128 | 45,585 | 45,457 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 128 | 1,152 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (8192B) | DMA_sram_xDMA↔C3_xDMA | 128 | 256 | 128 | ceil(8192/64)=128 |
| 7 | router [4,2048,64] | C3_VC | 1,152 | 3,213 | 2,061 | gemm(4,2048,64,[2x8x16])=2061 util=100% |
| 8 | topK | Host | 3,213 | 8,213 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 8,213 | 13,213 | 5,000 | ~5000cc |
| 10 | softmax | Host | 13,213 | 28,213 | 15,000 | ~15000cc |
| 11 | token_A→C2 (8192B) | SRAM(xDMA)→C2 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [xDMA] |
| 12 | token_A→C3 (8192B) | SRAM(iDMA)→C3 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [iDMA] |
| 13 | E0 gate+up [6,2048,1408] stream (dual-VC | C2_VC | 13,341 | 81,194 | 67,853 | pertile: 264tiles tile0=257 pipe=257 dma_total=22528 bank_s=1.00 →67853 [compute-bound] |
| 14 | C1 SiLU (11264 elem) | C1_elem | 45,541 | 45,629 | 88 | ceil(11264/128)=88 |
| 15 | C1 GLU (11264 elem) | C1_elem | 45,629 | 45,717 | 88 | ceil(11264/128)=88 |
| 16 | active_A C1→C0 (11264B) | DMA_C1_xDMA↔C0_xDMA | 45,717 | 45,893 | 176 | ceil(11264/64)=176 |
| 17 | C1 half_down [4,2816,1024] | C1_VC | 45,717 | 68,378 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 18 | C0 half_down [4,2816,1024] | C0_VC | 45,761 | 68,422 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 19 | merge half_down (4096B) | DMA_C1_xDMA↔C0_xDMA | 68,422 | 68,486 | 64 | ceil(4096/64)=64 |
| 20 | E0 SwiGLU | C2_elem | 81,194 | 81,260 | 66 | ceil(8448/128)=66 |
| 21 | E0 down [6,1408,1024]×2 stream (dual-VC  | C2_VC | 81,260 | 115,249 | 33,989 | pertile: 192tiles tile0=177 pipe=177 dma_total=11264 bank_s=1.00 →33989 [compute-bound] |
| 22 | E1 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 92,524 | 115,058 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 23 | E1 SwiGLU | C3_elem | 115,058 | 115,069 | 11 | ceil(1408/128)=11 |
| 24 | E1 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 115,069 | 126,339 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 25 | E2 gate+up [1,2048,1408] resident | C3_VC | 126,339 | 137,652 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 26 | E2 SwiGLU | C3_elem | 137,652 | 137,663 | 11 | ceil(1408/128)=11 |
| 27 | E2 down [1,1408,1024]×2 resident | C3_VC | 137,663 | 143,332 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |

#### 调度决策表 (M=4, 策略=event_driven)

- Token分布: 3experts: [6, 1, 1]
- Routed CC: 143,332, Shared CC: 68,486, Ratio: 2.093
- VC利用率: 89.0%, xDMA利用率: 33.2%, iDMA利用率: 33.2%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 6 | C2 | [2x8x16] | both | 128 | 1 | 否 | 100% | 101,908 | ED单发@128 6tok |
| E1 | 1 | C3 | [1x8x32] | both | 128 | 1 | 否 | 50% | 33,815 | ED单发@128 1tok |
| E2 | 1 | C3 | [1x8x32] | none | 0 | 1 | 是 | 100% | 16,993 | ED缓存 1tok |

---

**M=4 最差案例** (ratio=3.405)

### M=4 任务流表 (dist: 8experts: [1, 1, 1, 1, 1, 1, 1, 1])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 128 |  |  |  |  |  |  |  |  |  | token_A→C0 (8192B) |  |  |  |  | token_A→C1 (8192B) |  |
| 128 | 256 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (8192B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 256 | 1,152 | 896 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,152 | 3,213 | 2,061 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  | router [4,2048,64] |  |  |  |  |  |  |  |  |  |
| 3,213 | 8,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 8,213 | 13,213 | 5,000 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 13,213 | 13,341 | 128 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (8192B) | token_A→C2 (8192B) |  |  |
| 13,341 | 24,654 | 11,313 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 24,654 | 24,665 | 11 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  |  | E0 SwiGLU |  | E1 SwiGLU |  |  |  | softmax |  |  |  |  |
| 24,665 | 28,213 | 3,548 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  | softmax |  |  |  |  |
| 28,213 | 30,334 | 2,121 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 30,334 | 45,541 | 15,207 | C0 up_proj [4,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [4,2048,2816] |  | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,541 | 45,585 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (11264 elem) | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,585 | 45,629 | 44 |  |  |  | C1 SiLU (11264 elem) | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,629 | 45,717 | 88 |  |  |  | C1 GLU (11264 elem) | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 45,717 | 45,761 | 44 |  |  | C1 half_down [4,2816,1024] |  | E2 gate+up [1,2048,1408] strea |  |  |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,761 | 45,893 | 132 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 gate+up [1,2048,1408] strea |  |  |  | active_A C1→C0 (11264B) |  |  |  |  |  |  |  |
| 45,893 | 52,868 | 6,975 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 52,868 | 52,879 | 11 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 52,879 | 64,143 | 11,264 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 64,143 | 64,149 | 6 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  | E2 down [1,1408,1024]×2 stream |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 64,149 | 68,378 | 4,229 | C0 half_down [4,2816,1024] |  | C1 half_down [4,2816,1024] |  |  |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 68,378 | 68,422 | 44 | C0 half_down [4,2816,1024] |  |  |  |  |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 68,422 | 68,486 | 64 |  |  |  |  |  |  | E3 gate+up [1,2048,1408] strea |  | merge half_down (4096B) |  |  |  |  |  |  |  |
| 68,486 | 86,677 | 18,191 |  |  |  |  |  |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 86,677 | 86,688 | 11 |  |  |  |  |  |  |  | E3 SwiGLU |  |  |  |  |  |  |  |  |
| 86,688 | 97,952 | 11,264 |  |  |  |  |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 97,952 | 97,958 | 6 |  |  |  |  | E4 gate+up [1,2048,1408] strea |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 97,958 | 120,486 | 22,528 |  |  |  |  | E4 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 120,486 | 120,497 | 11 |  |  |  |  |  | E4 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 120,497 | 131,761 | 11,264 |  |  |  |  | E4 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 131,761 | 131,767 | 6 |  |  |  |  | E4 down [1,1408,1024]×2 stream |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 131,767 | 154,295 | 22,528 |  |  |  |  |  |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 154,295 | 154,306 | 11 |  |  |  |  |  |  |  | E5 SwiGLU |  |  |  |  |  |  |  |  |
| 154,306 | 165,570 | 11,264 |  |  |  |  |  |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 165,570 | 165,576 | 6 |  |  |  |  | E6 gate+up [1,2048,1408] strea |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 165,576 | 188,104 | 22,528 |  |  |  |  | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 188,104 | 188,115 | 11 |  |  |  |  |  | E6 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 188,115 | 199,379 | 11,264 |  |  |  |  | E6 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 199,379 | 199,385 | 6 |  |  |  |  | E6 down [1,1408,1024]×2 stream |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 199,385 | 221,913 | 22,528 |  |  |  |  |  |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 221,913 | 221,924 | 11 |  |  |  |  |  |  |  | E7 SwiGLU |  |  |  |  |  |  |  |  |
| 221,924 | 233,194 | 11,270 |  |  |  |  |  |  | E7 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=4)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 68,486 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 68,486 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 64,149 | C2 | E2_weights:4.125MB | 4.125MB | 0.875MB |
| 97,958 | C3 | E3_weights:4.125MB | 4.125MB | 0.875MB |
| 131,767 | C2 | E4_weights:4.125MB | 4.125MB | 0.875MB |
| 165,576 | C3 | E5_weights:4.125MB | 4.125MB | 0.875MB |
| 199,385 | C2 | E6_weights:4.125MB | 4.125MB | 0.875MB |
| 233,194 | C3 | E7_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=4)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (8192B) | DMA_sram_xDMA↔C0_xDMA | 0 | 128 | 128 | ceil(8192/64)=128 |
| 1 | token_A→C1 (8192B) | iDMA→C1 | 0 | 128 | 128 | ceil(8192/64)=128 |
| 2 | C0 up_proj [4,2048,2816] | C0_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 3 | C1 gate_proj [4,2048,2816] | C1_VC | 128 | 45,541 | 45,413 | gemm(4,2048,2816,[1x8x64])=45413 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 128 | 45,585 | 45,457 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 128 | 1,152 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (8192B) | DMA_sram_xDMA↔C3_xDMA | 128 | 256 | 128 | ceil(8192/64)=128 |
| 7 | router [4,2048,64] | C3_VC | 1,152 | 3,213 | 2,061 | gemm(4,2048,64,[2x8x16])=2061 util=100% |
| 8 | topK | Host | 3,213 | 8,213 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 8,213 | 13,213 | 5,000 | ~5000cc |
| 10 | softmax | Host | 13,213 | 28,213 | 15,000 | ~15000cc |
| 11 | token_A→C2 (8192B) | SRAM(xDMA)→C2 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [xDMA] |
| 12 | token_A→C3 (8192B) | SRAM(iDMA)→C3 | 13,213 | 13,341 | 128 | ceil(8192/64)=128 [iDMA] |
| 13 | E0 gate+up [1,2048,1408] resident | C2_VC | 13,341 | 24,654 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 14 | E1 gate+up [1,2048,1408] resident | C3_VC | 13,341 | 24,654 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 15 | E0 SwiGLU | C2_elem | 24,654 | 24,665 | 11 | ceil(1408/128)=11 |
| 16 | E1 SwiGLU | C3_elem | 24,654 | 24,665 | 11 | ceil(1408/128)=11 |
| 17 | E0 down [1,1408,1024]×2 resident | C2_VC | 24,665 | 30,334 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 18 | E1 down [1,1408,1024]×2 resident | C3_VC | 24,665 | 30,334 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 19 | E2 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 30,334 | 52,868 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 20 | C1 SiLU (11264 elem) | C1_elem | 45,541 | 45,629 | 88 | ceil(11264/128)=88 |
| 21 | C1 GLU (11264 elem) | C1_elem | 45,629 | 45,717 | 88 | ceil(11264/128)=88 |
| 22 | active_A C1→C0 (11264B) | DMA_C1_xDMA↔C0_xDMA | 45,717 | 45,893 | 176 | ceil(11264/64)=176 |
| 23 | C1 half_down [4,2816,1024] | C1_VC | 45,717 | 68,378 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 24 | C0 half_down [4,2816,1024] | C0_VC | 45,761 | 68,422 | 22,661 | gemm(4,2816,1024,[1x8x64])=22661 util=100% |
| 25 | E2 SwiGLU | C2_elem | 52,868 | 52,879 | 11 | ceil(1408/128)=11 |
| 26 | E2 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 52,879 | 64,149 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 27 | E3 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 64,143 | 86,677 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 28 | merge half_down (4096B) | DMA_C1_xDMA↔C0_xDMA | 68,422 | 68,486 | 64 | ceil(4096/64)=64 |
| 29 | E3 SwiGLU | C3_elem | 86,677 | 86,688 | 11 | ceil(1408/128)=11 |
| 30 | E3 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 86,688 | 97,958 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 31 | E4 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 97,952 | 120,486 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 32 | E4 SwiGLU | C2_elem | 120,486 | 120,497 | 11 | ceil(1408/128)=11 |
| 33 | E4 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 120,497 | 131,767 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 34 | E5 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 131,761 | 154,295 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 35 | E5 SwiGLU | C3_elem | 154,295 | 154,306 | 11 | ceil(1408/128)=11 |
| 36 | E5 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 154,306 | 165,576 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 37 | E6 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 165,570 | 188,104 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 38 | E6 SwiGLU | C2_elem | 188,104 | 188,115 | 11 | ceil(1408/128)=11 |
| 39 | E6 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 188,115 | 199,385 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 40 | E7 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 199,379 | 221,913 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 41 | E7 SwiGLU | C3_elem | 221,913 | 221,924 | 11 | ceil(1408/128)=11 |
| 42 | E7 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 221,924 | 233,194 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |

#### 调度决策表 (M=4, 策略=sequential_full)

- Token分布: 8experts: [1, 1, 1, 1, 1, 1, 1, 1]
- Routed CC: 233,194, Shared CC: 68,486, Ratio: 3.405
- VC利用率: 57.4%, xDMA利用率: 100.0%, iDMA利用率: 100.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E1 | 1 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C3 (省75%) |
| E2 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E3 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E4 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E5 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E6 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E7 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |

---

**M=8 最优案例** (ratio=1.107)

### M=8 任务流表 (dist: 1experts: [16])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 256 |  |  |  |  |  |  |  |  |  | token_A→C0 (16384B) |  |  |  |  | token_A→C1 (16384B) |  |
| 256 | 512 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (16384B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 512 | 1,280 | 768 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,280 | 5,397 | 4,117 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  | router [8,2048,64] |  |  |  |  |  |  |  |  |  |
| 5,397 | 10,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 10,397 | 15,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 15,397 | 15,653 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (16384B) | token_A→C2 (16384B) |  |  |
| 15,653 | 30,397 | 14,744 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  | softmax |  |  |  |  |
| 30,397 | 91,077 | 60,680 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 91,077 | 91,121 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (22528 elem) | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 91,121 | 91,253 | 132 |  |  |  | C1 SiLU (22528 elem) | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 91,253 | 91,429 | 176 |  |  |  | C1 GLU (22528 elem) | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 91,429 | 91,473 | 44 |  |  | C1 half_down [8,2816,1024] |  | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,473 | 91,781 | 308 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,781 | 106,122 | 14,341 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [8,2048,1408] strea |  | E0 gate+up [8,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 106,122 | 106,210 | 88 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  |  | E0 SwiGLU |  | E0 SwiGLU |  |  |  |  |  |  |  |  |
| 106,210 | 136,746 | 30,536 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 down [8,1408,1024]×2 stream |  | E0 down [8,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 136,746 | 136,790 | 44 | C0 half_down [8,2816,1024] |  |  |  | E0 down [8,1408,1024]×2 stream |  | E0 down [8,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 136,790 | 136,918 | 128 |  |  |  |  | E0 down [8,1408,1024]×2 stream |  | E0 down [8,1408,1024]×2 stream |  | merge half_down (8192B) |  |  |  |  |  |  |  |
| 136,918 | 151,527 | 14,609 |  |  |  |  | E0 down [8,1408,1024]×2 stream |  | E0 down [8,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=8)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 136,918 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 136,918 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 151,527 | C2 | E0_weights:4.125MB | 4.125MB | 0.875MB |
| 151,527 | C3 | E0_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=8)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (16384B) | DMA_sram_xDMA↔C0_xDMA | 0 | 256 | 256 | ceil(16384/64)=256 |
| 1 | token_A→C1 (16384B) | iDMA→C1 | 0 | 256 | 256 | ceil(16384/64)=256 |
| 2 | C0 up_proj [8,2048,2816] | C0_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 3 | C1 gate_proj [8,2048,2816] | C1_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 256 | 91,121 | 90,865 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 256 | 1,280 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (16384B) | DMA_sram_xDMA↔C3_xDMA | 256 | 512 | 256 | ceil(16384/64)=256 |
| 7 | router [8,2048,64] | C3_VC | 1,280 | 5,397 | 4,117 | gemm(8,2048,64,[2x8x16])=4117 util=100% |
| 8 | topK | Host | 5,397 | 10,397 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 10,397 | 15,397 | 5,000 | ~5000cc |
| 10 | softmax | Host | 15,397 | 30,397 | 15,000 | ~15000cc |
| 11 | token_A→C2 (16384B) | SRAM(xDMA)→C2 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [xDMA] |
| 12 | token_A→C3 (16384B) | SRAM(iDMA)→C3 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [iDMA] |
| 13 | E0 gate+up [8,2048,1408] stream (dual-VC | C2_VC | 15,653 | 106,122 | 90,469 | pertile: 352tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →90469 [compute-bound] |
| 14 | E0 gate+up [8,2048,1408] stream (dual-VC | C3_VC | 15,653 | 106,122 | 90,469 | pertile: 352tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →90469 [compute-bound] |
| 15 | C1 SiLU (22528 elem) | C1_elem | 91,077 | 91,253 | 176 | ceil(22528/128)=176 |
| 16 | C1 GLU (22528 elem) | C1_elem | 91,253 | 91,429 | 176 | ceil(22528/128)=176 |
| 17 | active_A C1→C0 (22528B) | DMA_C1_xDMA↔C0_xDMA | 91,429 | 91,781 | 352 | ceil(22528/64)=352 |
| 18 | C1 half_down [8,2816,1024] | C1_VC | 91,429 | 136,746 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 19 | C0 half_down [8,2816,1024] | C0_VC | 91,473 | 136,790 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 20 | E0 SwiGLU | C2_elem | 106,122 | 106,210 | 88 | ceil(11264/128)=88 |
| 21 | E0 SwiGLU | C3_elem | 106,122 | 106,210 | 88 | ceil(11264/128)=88 |
| 22 | E0 down [8,1408,1024]×2 stream (dual-VC  | C2_VC | 106,210 | 151,527 | 45,317 | pertile: 256tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →45317 [compute-bound] |
| 23 | E0 down [8,1408,1024]×2 stream (dual-VC  | C3_VC | 106,210 | 151,527 | 45,317 | pertile: 256tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →45317 [compute-bound] |
| 24 | merge half_down (8192B) | DMA_C1_xDMA↔C0_xDMA | 136,790 | 136,918 | 128 | ceil(8192/64)=128 |

#### 调度决策表 (M=8, 策略=unified_dynamic)

- Token分布: 1experts: [16]
- Routed CC: 151,527, Shared CC: 136,918, Ratio: 1.107
- VC利用率: 100.0%, xDMA利用率: 100.0%, iDMA利用率: 100.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 8 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 100% | 135,874 | 克隆模式: 16tok→8+8, C2@xdma64 |
| E0 | 8 | C3 | [4x8x8] | idma | 64 | 1 | 否 | 100% | 135,874 | 克隆模式: 16tok→8+8, C3@idma64 |

---

**M=8 中位案例** (ratio=1.603)

### M=8 任务流表 (dist: 5experts: [12, 1, 1, 1, 1])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 256 |  |  |  |  |  |  |  |  |  | token_A→C0 (16384B) |  |  |  |  | token_A→C1 (16384B) |  |
| 256 | 512 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (16384B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 512 | 1,280 | 768 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,280 | 5,397 | 4,117 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  | router [8,2048,64] |  |  |  |  |  |  |  |  |  |
| 5,397 | 10,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 10,397 | 15,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 15,397 | 15,653 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (16384B) | token_A→C2 (16384B) |  |  |
| 15,653 | 30,397 | 14,744 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [12,2048,1408] stre |  | E1 gate+up [1,2048,1408] strea |  |  |  |  | softmax |  |  |  |  |
| 30,397 | 60,715 | 30,318 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [12,2048,1408] stre |  | E1 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 60,715 | 60,726 | 11 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [12,2048,1408] stre |  |  | E1 SwiGLU |  |  |  |  |  |  |  |  |
| 60,726 | 83,260 | 22,534 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [12,2048,1408] stre |  | E1 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 83,260 | 91,077 | 7,817 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 91,077 | 91,121 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (22528 elem) | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 91,121 | 91,253 | 132 |  |  |  | C1 SiLU (22528 elem) | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 91,253 | 91,429 | 176 |  |  |  | C1 GLU (22528 elem) | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 91,429 | 91,473 | 44 |  |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,473 | 91,781 | 308 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,781 | 94,573 | 2,792 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  | E2 gate+up [1,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 94,573 | 94,584 | 11 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |
| 94,584 | 100,253 | 5,669 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  | E2 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 100,253 | 136,746 | 36,493 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E0 gate+up [12,2048,1408] stre |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 136,746 | 136,790 | 44 | C0 half_down [8,2816,1024] |  |  |  | E0 gate+up [12,2048,1408] stre |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 136,790 | 136,918 | 128 |  |  |  |  | E0 gate+up [12,2048,1408] stre |  | E3 gate+up [1,2048,1408] strea |  | merge half_down (8192B) |  |  |  |  |  |  |  |
| 136,918 | 145,315 | 8,397 |  |  |  |  | E0 gate+up [12,2048,1408] stre |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 145,315 | 145,326 | 11 |  |  |  |  | E0 gate+up [12,2048,1408] stre |  |  | E3 SwiGLU |  |  |  |  |  |  |  |  |
| 145,326 | 151,354 | 6,028 |  |  |  |  | E0 gate+up [12,2048,1408] stre |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 151,354 | 151,486 | 132 |  |  |  |  |  | E0 SwiGLU | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 151,486 | 167,860 | 16,374 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 167,860 | 174,014 | 6,154 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 174,014 | 196,548 | 22,534 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  | E4 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 196,548 | 196,559 | 11 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  |  | E4 SwiGLU |  |  |  |  |  |  |  |  |
| 196,559 | 207,829 | 11,270 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  | E4 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 207,829 | 219,459 | 11,630 |  |  |  |  | E0 down [12,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=8)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 136,918 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 136,918 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 83,260 | C3 | E1_weights:4.125MB | 4.125MB | 0.875MB |
| 219,459 | C2 | E0_weights:4.125MB | 4.125MB | 0.875MB |
| 167,860 | C3 | E3_weights:4.125MB | 4.125MB | 0.875MB |
| 207,829 | C3 | E4_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=8)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (16384B) | DMA_sram_xDMA↔C0_xDMA | 0 | 256 | 256 | ceil(16384/64)=256 |
| 1 | token_A→C1 (16384B) | iDMA→C1 | 0 | 256 | 256 | ceil(16384/64)=256 |
| 2 | C0 up_proj [8,2048,2816] | C0_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 3 | C1 gate_proj [8,2048,2816] | C1_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 256 | 91,121 | 90,865 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 256 | 1,280 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (16384B) | DMA_sram_xDMA↔C3_xDMA | 256 | 512 | 256 | ceil(16384/64)=256 |
| 7 | router [8,2048,64] | C3_VC | 1,280 | 5,397 | 4,117 | gemm(8,2048,64,[2x8x16])=4117 util=100% |
| 8 | topK | Host | 5,397 | 10,397 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 10,397 | 15,397 | 5,000 | ~5000cc |
| 10 | softmax | Host | 15,397 | 30,397 | 15,000 | ~15000cc |
| 11 | token_A→C2 (16384B) | SRAM(xDMA)→C2 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [xDMA] |
| 12 | token_A→C3 (16384B) | SRAM(iDMA)→C3 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [iDMA] |
| 13 | E1 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 15,653 | 60,715 | 45,062 | pertile: 44tiles tile0=1025 pipe=1024 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 14 | E0 gate+up [12,2048,1408] stream (dual-V | C2_VC | 15,653 | 151,354 | 135,701 | pertile: 528tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →135701 [compute-bound] |
| 15 | E1 SwiGLU | C3_elem | 60,715 | 60,726 | 11 | ceil(1408/128)=11 |
| 16 | E1 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 60,726 | 83,260 | 22,534 | pertile: 32tiles tile0=705 pipe=704 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 17 | E2 gate+up [1,2048,1408] resident | C3_VC | 83,260 | 94,573 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 18 | C1 SiLU (22528 elem) | C1_elem | 91,077 | 91,253 | 176 | ceil(22528/128)=176 |
| 19 | C1 GLU (22528 elem) | C1_elem | 91,253 | 91,429 | 176 | ceil(22528/128)=176 |
| 20 | active_A C1→C0 (22528B) | DMA_C1_xDMA↔C0_xDMA | 91,429 | 91,781 | 352 | ceil(22528/64)=352 |
| 21 | C1 half_down [8,2816,1024] | C1_VC | 91,429 | 136,746 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 22 | C0 half_down [8,2816,1024] | C0_VC | 91,473 | 136,790 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 23 | E2 SwiGLU | C3_elem | 94,573 | 94,584 | 11 | ceil(1408/128)=11 |
| 24 | E2 down [1,1408,1024]×2 resident | C3_VC | 94,584 | 100,253 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 25 | E3 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 100,253 | 145,315 | 45,062 | pertile: 44tiles tile0=1025 pipe=1024 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 26 | merge half_down (8192B) | DMA_C1_xDMA↔C0_xDMA | 136,790 | 136,918 | 128 | ceil(8192/64)=128 |
| 27 | E3 SwiGLU | C3_elem | 145,315 | 145,326 | 11 | ceil(1408/128)=11 |
| 28 | E3 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 145,326 | 167,860 | 22,534 | pertile: 32tiles tile0=705 pipe=704 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 29 | E0 SwiGLU | C2_elem | 151,354 | 151,486 | 132 | ceil(16896/128)=132 |
| 30 | E0 down [12,1408,1024]×2 stream (dual-VC | C2_VC | 151,486 | 219,459 | 67,973 | pertile: 384tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →67973 [compute-bound] |
| 31 | E4 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 174,014 | 196,548 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 32 | E4 SwiGLU | C3_elem | 196,548 | 196,559 | 11 | ceil(1408/128)=11 |
| 33 | E4 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 196,559 | 207,829 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |

#### 调度决策表 (M=8, 策略=event_driven)

- Token分布: 5experts: [12, 1, 1, 1, 1]
- Routed CC: 219,459, Shared CC: 136,918, Ratio: 1.603
- VC利用率: 69.7%, xDMA利用率: 74.6%, iDMA利用率: 41.5%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E1 | 1 | C3 | [1x8x32] | xdma | 64 | 1 | 否 | 25% | 67,607 | ED单发@64 1tok |
| E0 | 12 | C2 | [4x8x8] | idma | 64 | 1 | 否 | 100% | 203,806 | ED单发@64 12tok |
| E2 | 1 | C3 | [1x8x32] | none | 0 | 1 | 是 | 100% | 16,993 | ED缓存 1tok |
| E3 | 1 | C3 | [1x8x32] | xdma | 64 | 1 | 否 | 25% | 67,607 | ED单发@64 1tok |
| E4 | 1 | C3 | [1x8x32] | both | 128 | 1 | 否 | 50% | 33,815 | ED单发@128 1tok |

---

**M=8 最差案例** (ratio=3.695)

### M=8 任务流表 (dist: 16experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(6 more))

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 256 |  |  |  |  |  |  |  |  |  | token_A→C0 (16384B) |  |  |  |  | token_A→C1 (16384B) |  |
| 256 | 512 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (16384B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 512 | 1,280 | 768 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,280 | 5,397 | 4,117 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  | router [8,2048,64] |  |  |  |  |  |  |  |  |  |
| 5,397 | 10,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 10,397 | 15,397 | 5,000 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 15,397 | 15,653 | 256 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (16384B) | token_A→C2 (16384B) |  |  |
| 15,653 | 26,966 | 11,313 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 26,966 | 26,977 | 11 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  | E0 SwiGLU |  | E1 SwiGLU |  |  |  | softmax |  |  |  |  |
| 26,977 | 30,397 | 3,420 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  | softmax |  |  |  |  |
| 30,397 | 32,646 | 2,249 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 32,646 | 55,180 | 22,534 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 55,180 | 55,191 | 11 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 55,191 | 66,455 | 11,264 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E2 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 66,455 | 66,461 | 6 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  | E2 down [1,1408,1024]×2 stream |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 66,461 | 88,989 | 22,528 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 88,989 | 89,000 | 11 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  |  | E3 SwiGLU |  |  |  |  |  |  |  |  |
| 89,000 | 91,077 | 2,077 | C0 up_proj [8,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [8,2048,2816] |  |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 91,077 | 91,121 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (22528 elem) |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 91,121 | 91,253 | 132 |  |  |  | C1 SiLU (22528 elem) |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 91,253 | 91,429 | 176 |  |  |  | C1 GLU (22528 elem) |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 91,429 | 91,473 | 44 |  |  | C1 half_down [8,2816,1024] |  |  |  | E3 down [1,1408,1024]×2 stream |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,473 | 91,781 | 308 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  |  |  | E3 down [1,1408,1024]×2 stream |  | active_A C1→C0 (22528B) |  |  |  |  |  |  |  |
| 91,781 | 100,264 | 8,483 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 100,264 | 100,270 | 6 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E4 gate+up [1,2048,1408] strea |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 100,270 | 122,798 | 22,528 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E4 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 122,798 | 122,809 | 11 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  |  | E4 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 122,809 | 134,073 | 11,264 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E4 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 134,073 | 134,079 | 6 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  | E4 down [1,1408,1024]×2 stream |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 134,079 | 136,746 | 2,667 | C0 half_down [8,2816,1024] |  | C1 half_down [8,2816,1024] |  |  |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 136,746 | 136,790 | 44 | C0 half_down [8,2816,1024] |  |  |  |  |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 136,790 | 136,918 | 128 |  |  |  |  |  |  | E5 gate+up [1,2048,1408] strea |  | merge half_down (8192B) |  |  |  |  |  |  |  |
| 136,918 | 156,607 | 19,689 |  |  |  |  |  |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 156,607 | 156,618 | 11 |  |  |  |  |  |  |  | E5 SwiGLU |  |  |  |  |  |  |  |  |
| 156,618 | 167,882 | 11,264 |  |  |  |  |  |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 167,882 | 167,888 | 6 |  |  |  |  | E6 gate+up [1,2048,1408] strea |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 167,888 | 190,416 | 22,528 |  |  |  |  | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 190,416 | 190,427 | 11 |  |  |  |  |  | E6 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 190,427 | 201,691 | 11,264 |  |  |  |  | E6 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 201,691 | 201,697 | 6 |  |  |  |  | E6 down [1,1408,1024]×2 stream |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 201,697 | 224,225 | 22,528 |  |  |  |  |  |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 224,225 | 224,236 | 11 |  |  |  |  |  |  |  | E7 SwiGLU |  |  |  |  |  |  |  |  |
| 224,236 | 235,500 | 11,264 |  |  |  |  |  |  | E7 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 235,500 | 235,506 | 6 |  |  |  |  | E8 gate+up [1,2048,1408] strea |  | E7 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 235,506 | 258,034 | 22,528 |  |  |  |  | E8 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 258,034 | 258,045 | 11 |  |  |  |  |  | E8 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 258,045 | 269,309 | 11,264 |  |  |  |  | E8 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 269,309 | 269,315 | 6 |  |  |  |  | E8 down [1,1408,1024]×2 stream |  | E9 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 269,315 | 291,843 | 22,528 |  |  |  |  |  |  | E9 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 291,843 | 291,854 | 11 |  |  |  |  |  |  |  | E9 SwiGLU |  |  |  |  |  |  |  |  |
| 291,854 | 303,118 | 11,264 |  |  |  |  |  |  | E9 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 303,118 | 303,124 | 6 |  |  |  |  | E10 gate+up [1,2048,1408] stre |  | E9 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 303,124 | 325,652 | 22,528 |  |  |  |  | E10 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 325,652 | 325,663 | 11 |  |  |  |  |  | E10 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 325,663 | 336,927 | 11,264 |  |  |  |  | E10 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 336,927 | 336,933 | 6 |  |  |  |  | E10 down [1,1408,1024]×2 strea |  | E11 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 336,933 | 359,461 | 22,528 |  |  |  |  |  |  | E11 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 359,461 | 359,472 | 11 |  |  |  |  |  |  |  | E11 SwiGLU |  |  |  |  |  |  |  |  |
| 359,472 | 370,736 | 11,264 |  |  |  |  |  |  | E11 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 370,736 | 370,742 | 6 |  |  |  |  | E12 gate+up [1,2048,1408] stre |  | E11 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 370,742 | 393,270 | 22,528 |  |  |  |  | E12 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 393,270 | 393,281 | 11 |  |  |  |  |  | E12 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 393,281 | 404,545 | 11,264 |  |  |  |  | E12 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 404,545 | 404,551 | 6 |  |  |  |  | E12 down [1,1408,1024]×2 strea |  | E13 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 404,551 | 427,079 | 22,528 |  |  |  |  |  |  | E13 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 427,079 | 427,090 | 11 |  |  |  |  |  |  |  | E13 SwiGLU |  |  |  |  |  |  |  |  |
| 427,090 | 438,354 | 11,264 |  |  |  |  |  |  | E13 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 438,354 | 438,360 | 6 |  |  |  |  | E14 gate+up [1,2048,1408] stre |  | E13 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 438,360 | 460,888 | 22,528 |  |  |  |  | E14 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 460,888 | 460,899 | 11 |  |  |  |  |  | E14 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 460,899 | 472,163 | 11,264 |  |  |  |  | E14 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 472,163 | 472,169 | 6 |  |  |  |  | E14 down [1,1408,1024]×2 strea |  | E15 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 472,169 | 494,697 | 22,528 |  |  |  |  |  |  | E15 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 494,697 | 494,708 | 11 |  |  |  |  |  |  |  | E15 SwiGLU |  |  |  |  |  |  |  |  |
| 494,708 | 505,978 | 11,270 |  |  |  |  |  |  | E15 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=8)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 136,918 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 136,918 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 66,461 | C2 | E2_weights:4.125MB | 4.125MB | 0.875MB |
| 100,270 | C3 | E3_weights:4.125MB | 4.125MB | 0.875MB |
| 134,079 | C2 | E4_weights:4.125MB | 4.125MB | 0.875MB |
| 167,888 | C3 | E5_weights:4.125MB | 4.125MB | 0.875MB |
| 201,697 | C2 | E6_weights:4.125MB | 4.125MB | 0.875MB |
| 235,506 | C3 | E7_weights:4.125MB | 4.125MB | 0.875MB |
| 269,315 | C2 | E8_weights:4.125MB | 4.125MB | 0.875MB |
| 303,124 | C3 | E9_weights:4.125MB | 4.125MB | 0.875MB |
| 336,933 | C2 | E10_weights:4.125MB | 4.125MB | 0.875MB |
| 370,742 | C3 | E11_weights:4.125MB | 4.125MB | 0.875MB |
| 404,551 | C2 | E12_weights:4.125MB | 4.125MB | 0.875MB |
| 438,360 | C3 | E13_weights:4.125MB | 4.125MB | 0.875MB |
| 472,169 | C2 | E14_weights:4.125MB | 4.125MB | 0.875MB |
| 505,978 | C3 | E15_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=8)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (16384B) | DMA_sram_xDMA↔C0_xDMA | 0 | 256 | 256 | ceil(16384/64)=256 |
| 1 | token_A→C1 (16384B) | iDMA→C1 | 0 | 256 | 256 | ceil(16384/64)=256 |
| 2 | C0 up_proj [8,2048,2816] | C0_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 3 | C1 gate_proj [8,2048,2816] | C1_VC | 256 | 91,077 | 90,821 | gemm(8,2048,2816,[1x8x64])=90821 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 256 | 91,121 | 90,865 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 256 | 1,280 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (16384B) | DMA_sram_xDMA↔C3_xDMA | 256 | 512 | 256 | ceil(16384/64)=256 |
| 7 | router [8,2048,64] | C3_VC | 1,280 | 5,397 | 4,117 | gemm(8,2048,64,[2x8x16])=4117 util=100% |
| 8 | topK | Host | 5,397 | 10,397 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 10,397 | 15,397 | 5,000 | ~5000cc |
| 10 | softmax | Host | 15,397 | 30,397 | 15,000 | ~15000cc |
| 11 | token_A→C2 (16384B) | SRAM(xDMA)→C2 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [xDMA] |
| 12 | token_A→C3 (16384B) | SRAM(iDMA)→C3 | 15,397 | 15,653 | 256 | ceil(16384/64)=256 [iDMA] |
| 13 | E0 gate+up [1,2048,1408] resident | C2_VC | 15,653 | 26,966 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 14 | E1 gate+up [1,2048,1408] resident | C3_VC | 15,653 | 26,966 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 15 | E0 SwiGLU | C2_elem | 26,966 | 26,977 | 11 | ceil(1408/128)=11 |
| 16 | E1 SwiGLU | C3_elem | 26,966 | 26,977 | 11 | ceil(1408/128)=11 |
| 17 | E0 down [1,1408,1024]×2 resident | C2_VC | 26,977 | 32,646 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 18 | E1 down [1,1408,1024]×2 resident | C3_VC | 26,977 | 32,646 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 19 | E2 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 32,646 | 55,180 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 20 | E2 SwiGLU | C2_elem | 55,180 | 55,191 | 11 | ceil(1408/128)=11 |
| 21 | E2 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 55,191 | 66,461 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 22 | E3 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 66,455 | 88,989 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 23 | E3 SwiGLU | C3_elem | 88,989 | 89,000 | 11 | ceil(1408/128)=11 |
| 24 | E3 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 89,000 | 100,270 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 25 | C1 SiLU (22528 elem) | C1_elem | 91,077 | 91,253 | 176 | ceil(22528/128)=176 |
| 26 | C1 GLU (22528 elem) | C1_elem | 91,253 | 91,429 | 176 | ceil(22528/128)=176 |
| 27 | active_A C1→C0 (22528B) | DMA_C1_xDMA↔C0_xDMA | 91,429 | 91,781 | 352 | ceil(22528/64)=352 |
| 28 | C1 half_down [8,2816,1024] | C1_VC | 91,429 | 136,746 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 29 | C0 half_down [8,2816,1024] | C0_VC | 91,473 | 136,790 | 45,317 | gemm(8,2816,1024,[1x8x64])=45317 util=100% |
| 30 | E4 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 100,264 | 122,798 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 31 | E4 SwiGLU | C2_elem | 122,798 | 122,809 | 11 | ceil(1408/128)=11 |
| 32 | E4 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 122,809 | 134,079 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 33 | E5 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 134,073 | 156,607 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 34 | merge half_down (8192B) | DMA_C1_xDMA↔C0_xDMA | 136,790 | 136,918 | 128 | ceil(8192/64)=128 |
| 35 | E5 SwiGLU | C3_elem | 156,607 | 156,618 | 11 | ceil(1408/128)=11 |
| 36 | E5 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 156,618 | 167,888 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 37 | E6 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 167,882 | 190,416 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 38 | E6 SwiGLU | C2_elem | 190,416 | 190,427 | 11 | ceil(1408/128)=11 |
| 39 | E6 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 190,427 | 201,697 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 40 | E7 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 201,691 | 224,225 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 41 | E7 SwiGLU | C3_elem | 224,225 | 224,236 | 11 | ceil(1408/128)=11 |
| 42 | E7 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 224,236 | 235,506 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 43 | E8 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 235,500 | 258,034 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 44 | E8 SwiGLU | C2_elem | 258,034 | 258,045 | 11 | ceil(1408/128)=11 |
| 45 | E8 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 258,045 | 269,315 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 46 | E9 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 269,309 | 291,843 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 47 | E9 SwiGLU | C3_elem | 291,843 | 291,854 | 11 | ceil(1408/128)=11 |
| 48 | E9 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 291,854 | 303,124 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 49 | E10 gate+up [1,2048,1408] stream (dual-V | C2_VC | 303,118 | 325,652 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 50 | E10 SwiGLU | C2_elem | 325,652 | 325,663 | 11 | ceil(1408/128)=11 |
| 51 | E10 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 325,663 | 336,933 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 52 | E11 gate+up [1,2048,1408] stream (dual-V | C3_VC | 336,927 | 359,461 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 53 | E11 SwiGLU | C3_elem | 359,461 | 359,472 | 11 | ceil(1408/128)=11 |
| 54 | E11 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 359,472 | 370,742 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 55 | E12 gate+up [1,2048,1408] stream (dual-V | C2_VC | 370,736 | 393,270 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 56 | E12 SwiGLU | C2_elem | 393,270 | 393,281 | 11 | ceil(1408/128)=11 |
| 57 | E12 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 393,281 | 404,551 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 58 | E13 gate+up [1,2048,1408] stream (dual-V | C3_VC | 404,545 | 427,079 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 59 | E13 SwiGLU | C3_elem | 427,079 | 427,090 | 11 | ceil(1408/128)=11 |
| 60 | E13 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 427,090 | 438,360 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 61 | E14 gate+up [1,2048,1408] stream (dual-V | C2_VC | 438,354 | 460,888 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 62 | E14 SwiGLU | C2_elem | 460,888 | 460,899 | 11 | ceil(1408/128)=11 |
| 63 | E14 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 460,899 | 472,169 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 64 | E15 gate+up [1,2048,1408] stream (dual-V | C3_VC | 472,163 | 494,697 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 65 | E15 SwiGLU | C3_elem | 494,697 | 494,708 | 11 | ceil(1408/128)=11 |
| 66 | E15 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 494,708 | 505,978 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |

#### 调度决策表 (M=8, 策略=sequential_full)

- Token分布: 16experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(6 more)
- Routed CC: 505,978, Shared CC: 136,918, Ratio: 3.695
- VC利用率: 53.6%, xDMA利用率: 100.0%, iDMA利用率: 100.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E1 | 1 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C3 (省75%) |
| E2 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E3 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E4 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E5 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E6 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E7 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E8 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E9 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E10 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E11 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E12 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E13 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E14 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E15 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |

---

**M=16 最优案例** (ratio=0.881)

### M=16 任务流表 (dist: 4experts: [10, 10, 9, 3])

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 512 | 512 |  |  |  |  |  |  |  |  |  | token_A→C0 (32768B) |  |  |  |  | token_A→C1 (32768B) |  |
| 512 | 1,024 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (32768B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,024 | 1,536 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,536 | 9,765 | 8,229 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | router [16,2048,64] |  |  |  |  |  |  |  |  |  |
| 9,765 | 14,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 14,765 | 19,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 19,765 | 20,277 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (32768B) | token_A→C2 (32768B) |  |  |
| 20,277 | 34,765 | 14,488 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E1 gate+up [3,2048,1408] resid |  | E2 gate+up [9,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 34,765 | 54,206 | 19,441 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E1 gate+up [3,2048,1408] resid |  | E2 gate+up [9,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 54,206 | 54,239 | 33 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E1 SwiGLU | E2 gate+up [9,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 54,239 | 71,236 | 16,997 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E1 down [3,1408,1024]×2 reside |  | E2 gate+up [9,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 71,236 | 122,054 | 50,818 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [10,2048,1408] stre |  | E2 gate+up [9,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 122,054 | 122,153 | 99 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [10,2048,1408] stre |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |
| 122,153 | 173,134 | 50,981 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [10,2048,1408] stre |  | E2 down [9,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 173,134 | 182,149 | 9,015 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [10,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 182,149 | 182,193 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (45056 elem) | E0 gate+up [10,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 182,193 | 182,501 | 308 |  |  |  | C1 SiLU (45056 elem) | E0 gate+up [10,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 182,501 | 182,853 | 352 |  |  |  | C1 GLU (45056 elem) | E0 gate+up [10,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 182,853 | 182,897 | 44 |  |  | C1 half_down [16,2816,1024] |  | E0 gate+up [10,2048,1408] stre |  |  |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 182,897 | 183,557 | 660 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E0 gate+up [10,2048,1408] stre |  |  |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 183,557 | 184,321 | 764 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E0 gate+up [10,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 184,321 | 184,431 | 110 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  | E0 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 184,431 | 241,076 | 56,645 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E0 down [10,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 241,076 | 273,482 | 32,406 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 273,482 | 273,526 | 44 | C0 half_down [16,2816,1024] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 273,526 | 273,782 | 256 |  |  |  |  |  |  |  |  | merge half_down (16384B) |  |  |  |  |  |  |  |

#### TCDM状态 (M=16)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 273,782 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 273,782 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 241,076 | C2 | E0_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=16)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (32768B) | DMA_sram_xDMA↔C0_xDMA | 0 | 512 | 512 | ceil(32768/64)=512 |
| 1 | token_A→C1 (32768B) | iDMA→C1 | 0 | 512 | 512 | ceil(32768/64)=512 |
| 2 | C0 up_proj [16,2048,2816] | C0_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 3 | C1 gate_proj [16,2048,2816] | C1_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 512 | 182,193 | 181,681 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 512 | 1,536 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (32768B) | DMA_sram_xDMA↔C3_xDMA | 512 | 1,024 | 512 | ceil(32768/64)=512 |
| 7 | router [16,2048,64] | C3_VC | 1,536 | 9,765 | 8,229 | gemm(16,2048,64,[2x8x16])=8229 util=100% |
| 8 | topK | Host | 9,765 | 14,765 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 14,765 | 19,765 | 5,000 | ~5000cc |
| 10 | softmax | Host | 19,765 | 34,765 | 15,000 | ~15000cc |
| 11 | token_A→C2 (32768B) | SRAM(xDMA)→C2 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [xDMA] |
| 12 | token_A→C3 (32768B) | SRAM(iDMA)→C3 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [iDMA] |
| 13 | E2 gate+up [9,2048,1408] resident | C3_VC | 20,277 | 122,054 | 101,777 | dual_vc_gu_resident: gemm(9,2048,1408,[1x8x32])=101777 util=100% |
| 14 | E1 gate+up [3,2048,1408] resident | C2_VC | 20,277 | 54,206 | 33,929 | dual_vc_gu_resident: gemm(3,2048,1408,[1x8x32])=33929 util=100% |
| 15 | E1 SwiGLU | C2_elem | 54,206 | 54,239 | 33 | ceil(4224/128)=33 |
| 16 | E1 down [3,1408,1024]×2 resident | C2_VC | 54,239 | 71,236 | 16,997 | dual_vc_dn_resident: gemm(3,1408,1024,[1x8x32])=16997 util=100% |
| 17 | E0 gate+up [10,2048,1408] stream (dual-V | C2_VC | 71,236 | 184,321 | 113,085 | pertile: 440tiles tile0=257 pipe=257 dma_total=22528 bank_s=1.00 →113085 [compute-bound] |
| 18 | E2 SwiGLU | C3_elem | 122,054 | 122,153 | 99 | ceil(12672/128)=99 |
| 19 | E2 down [9,1408,1024]×2 resident | C3_VC | 122,153 | 173,134 | 50,981 | dual_vc_dn_resident: gemm(9,1408,1024,[1x8x32])=50981 util=100% |
| 20 | C1 SiLU (45056 elem) | C1_elem | 182,149 | 182,501 | 352 | ceil(45056/128)=352 |
| 21 | C1 GLU (45056 elem) | C1_elem | 182,501 | 182,853 | 352 | ceil(45056/128)=352 |
| 22 | active_A C1→C0 (45056B) | DMA_C1_xDMA↔C0_xDMA | 182,853 | 183,557 | 704 | ceil(45056/64)=704 |
| 23 | C1 half_down [16,2816,1024] | C1_VC | 182,853 | 273,482 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 24 | C0 half_down [16,2816,1024] | C0_VC | 182,897 | 273,526 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 25 | E0 SwiGLU | C2_elem | 184,321 | 184,431 | 110 | ceil(14080/128)=110 |
| 26 | E0 down [10,1408,1024]×2 stream (dual-VC | C2_VC | 184,431 | 241,076 | 56,645 | pertile: 320tiles tile0=177 pipe=177 dma_total=11264 bank_s=1.00 →56645 [compute-bound] |
| 27 | merge half_down (16384B) | DMA_C1_xDMA↔C0_xDMA | 273,526 | 273,782 | 256 | ceil(16384/64)=256 |

#### 调度决策表 (M=16, 策略=online_greedy)

- Token分布: 4experts: [10, 10, 9, 3]
- Routed CC: 241,076, Shared CC: 273,782, Ratio: 0.881
- VC利用率: 100.0%, xDMA利用率: 76.9%, iDMA利用率: 76.9%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E2 | 9 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 152,857 | 缓存命中 resident 9tok @C3 (省25%) |
| E1 | 3 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 50,959 | 缓存命中 resident 3tok @C2 (省25%) |
| E0 | 10 | C2 | [2x8x16] | both | 128 | 1 | 否 | 100% | 169,840 | online_greedy @128B/cc |

---

**M=16 中位案例** (ratio=1.434)

### M=16 任务流表 (dist: 12experts: [4, 4, 4, 3, 3, 3, 2, 2, 2, 2]...(2 more))

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 512 | 512 |  |  |  |  |  |  |  |  |  | token_A→C0 (32768B) |  |  |  |  | token_A→C1 (32768B) |  |
| 512 | 1,024 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (32768B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,024 | 1,536 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,536 | 9,765 | 8,229 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | router [16,2048,64] |  |  |  |  |  |  |  |  |  |
| 9,765 | 14,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 14,765 | 19,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 19,765 | 20,277 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (32768B) | token_A→C2 (32768B) |  |  |
| 20,277 | 31,590 | 11,313 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E6 gate+up [1,2048,1408] resid |  | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 31,590 | 31,601 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E6 SwiGLU | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 31,601 | 34,765 | 3,164 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E6 down [1,1408,1024]×2 reside |  | E1 gate+up [2,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 34,765 | 37,270 | 2,505 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E6 down [1,1408,1024]×2 reside |  | E1 gate+up [2,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 37,270 | 42,898 | 5,628 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [4,2048,1408] strea |  | E1 gate+up [2,2048,1408] resid |  |  |  |  |  |  |  |  |  |
| 42,898 | 42,920 | 22 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [4,2048,1408] strea |  |  | E1 SwiGLU |  |  |  |  |  |  |  |  |
| 42,920 | 54,253 | 11,333 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [4,2048,1408] strea |  | E1 down [2,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 54,253 | 82,507 | 28,254 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [4,2048,1408] strea |  | E5 gate+up [3,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 82,507 | 82,551 | 44 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E0 SwiGLU | E5 gate+up [3,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 82,551 | 99,490 | 16,939 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 down [4,1408,1024]×2 stream |  | E5 gate+up [3,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 99,490 | 99,523 | 33 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 down [4,1408,1024]×2 stream |  |  | E5 SwiGLU |  |  |  |  |  |  |  |  |
| 99,523 | 105,212 | 5,689 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 down [4,1408,1024]×2 stream |  | E5 down [3,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 105,212 | 122,184 | 16,972 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 gate+up [4,2048,1408] strea |  | E5 down [3,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 122,184 | 150,449 | 28,265 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 gate+up [4,2048,1408] strea |  | E3 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 150,449 | 150,493 | 44 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E2 SwiGLU | E3 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 150,493 | 167,246 | 16,753 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 down [4,1408,1024]×2 stream |  | E3 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 167,246 | 167,268 | 22 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 down [4,1408,1024]×2 stream |  |  | E3 SwiGLU |  |  |  |  |  |  |  |  |
| 167,268 | 173,154 | 5,886 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 down [4,1408,1024]×2 stream |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 173,154 | 182,149 | 8,995 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 182,149 | 182,193 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (45056 elem) | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 182,193 | 182,501 | 308 |  |  |  | C1 SiLU (45056 elem) | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 182,501 | 182,853 | 352 |  |  |  | C1 GLU (45056 elem) | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 182,853 | 182,897 | 44 |  |  | C1 half_down [16,2816,1024] |  | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 182,897 | 183,557 | 660 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 183,557 | 189,802 | 6,245 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 gate+up [4,2048,1408] strea |  | E3 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 189,802 | 218,391 | 28,589 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 gate+up [4,2048,1408] strea |  | E7 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 218,391 | 218,435 | 44 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  | E4 SwiGLU | E7 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 218,435 | 234,864 | 16,429 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 down [4,1408,1024]×2 stream |  | E7 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 234,864 | 234,886 | 22 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 down [4,1408,1024]×2 stream |  |  | E7 SwiGLU |  |  |  |  |  |  |  |  |
| 234,886 | 241,096 | 6,210 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E4 down [4,1408,1024]×2 stream |  | E7 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 241,096 | 257,420 | 16,324 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E8 gate+up [3,2048,1408] strea |  | E7 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 257,420 | 273,482 | 16,062 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E8 gate+up [3,2048,1408] strea |  | E9 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 273,482 | 273,526 | 44 | C0 half_down [16,2816,1024] |  |  |  | E8 gate+up [3,2048,1408] strea |  | E9 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 273,526 | 273,782 | 256 |  |  |  |  | E8 gate+up [3,2048,1408] strea |  | E9 gate+up [2,2048,1408] strea |  | merge half_down (16384B) |  |  |  |  |  |  |  |
| 273,782 | 286,333 | 12,551 |  |  |  |  | E8 gate+up [3,2048,1408] strea |  | E9 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 286,333 | 286,366 | 33 |  |  |  |  |  | E8 SwiGLU | E9 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 286,366 | 302,482 | 16,116 |  |  |  |  | E8 down [3,1408,1024]×2 stream |  | E9 gate+up [2,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 302,482 | 302,504 | 22 |  |  |  |  | E8 down [3,1408,1024]×2 stream |  |  | E9 SwiGLU |  |  |  |  |  |  |  |  |
| 302,504 | 309,027 | 6,523 |  |  |  |  | E8 down [3,1408,1024]×2 stream |  | E9 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 309,027 | 325,038 | 16,011 |  |  |  |  | E10 gate+up [3,2048,1408] stre |  | E9 down [2,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 325,038 | 354,264 | 29,226 |  |  |  |  | E10 gate+up [3,2048,1408] stre |  | E11 gate+up [2,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 354,264 | 354,297 | 33 |  |  |  |  |  | E10 SwiGLU | E11 gate+up [2,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 354,297 | 370,100 | 15,803 |  |  |  |  | E10 down [3,1408,1024]×2 strea |  | E11 gate+up [2,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 370,100 | 370,122 | 22 |  |  |  |  | E10 down [3,1408,1024]×2 strea |  |  | E11 SwiGLU |  |  |  |  |  |  |  |  |
| 370,122 | 376,958 | 6,836 |  |  |  |  | E10 down [3,1408,1024]×2 strea |  | E11 down [2,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 376,958 | 392,656 | 15,698 |  |  |  |  |  |  | E11 down [2,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=16)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 273,782 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 273,782 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 105,212 | C2 | E0_weights:4.125MB | 4.125MB | 0.875MB |
| 122,184 | C3 | E5_weights:4.125MB | 4.125MB | 0.875MB |
| 173,154 | C2 | E2_weights:4.125MB | 4.125MB | 0.875MB |
| 189,802 | C3 | E3_weights:4.125MB | 4.125MB | 0.875MB |
| 241,096 | C2 | E4_weights:4.125MB | 4.125MB | 0.875MB |
| 257,420 | C3 | E7_weights:4.125MB | 4.125MB | 0.875MB |
| 309,027 | C2 | E8_weights:4.125MB | 4.125MB | 0.875MB |
| 325,038 | C3 | E9_weights:4.125MB | 4.125MB | 0.875MB |
| 376,958 | C2 | E10_weights:4.125MB | 4.125MB | 0.875MB |
| 392,656 | C3 | E11_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=16)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (32768B) | DMA_sram_xDMA↔C0_xDMA | 0 | 512 | 512 | ceil(32768/64)=512 |
| 1 | token_A→C1 (32768B) | iDMA→C1 | 0 | 512 | 512 | ceil(32768/64)=512 |
| 2 | C0 up_proj [16,2048,2816] | C0_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 3 | C1 gate_proj [16,2048,2816] | C1_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 512 | 182,193 | 181,681 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 512 | 1,536 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (32768B) | DMA_sram_xDMA↔C3_xDMA | 512 | 1,024 | 512 | ceil(32768/64)=512 |
| 7 | router [16,2048,64] | C3_VC | 1,536 | 9,765 | 8,229 | gemm(16,2048,64,[2x8x16])=8229 util=100% |
| 8 | topK | Host | 9,765 | 14,765 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 14,765 | 19,765 | 5,000 | ~5000cc |
| 10 | softmax | Host | 19,765 | 34,765 | 15,000 | ~15000cc |
| 11 | token_A→C2 (32768B) | SRAM(xDMA)→C2 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [xDMA] |
| 12 | token_A→C3 (32768B) | SRAM(iDMA)→C3 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [iDMA] |
| 13 | E1 gate+up [2,2048,1408] resident | C3_VC | 20,277 | 42,898 | 22,621 | dual_vc_gu_resident: gemm(2,2048,1408,[1x8x32])=22621 util=100% |
| 14 | E6 gate+up [1,2048,1408] resident | C2_VC | 20,277 | 31,590 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 15 | E6 SwiGLU | C2_elem | 31,590 | 31,601 | 11 | ceil(1408/128)=11 |
| 16 | E6 down [1,1408,1024]×2 resident | C2_VC | 31,601 | 37,270 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 17 | E0 gate+up [4,2048,1408] stream (dual-VC | C2_VC | 37,270 | 82,507 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 18 | E1 SwiGLU | C3_elem | 42,898 | 42,920 | 22 | ceil(2816/128)=22 |
| 19 | E1 down [2,1408,1024]×2 resident | C3_VC | 42,920 | 54,253 | 11,333 | dual_vc_dn_resident: gemm(2,1408,1024,[1x8x32])=11333 util=100% |
| 20 | E5 gate+up [3,2048,1408] stream (dual-VC | C3_VC | 54,253 | 99,490 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 21 | E0 SwiGLU | C2_elem | 82,507 | 82,551 | 44 | ceil(5632/128)=44 |
| 22 | E0 down [4,1408,1024]×2 stream (dual-VC  | C2_VC | 82,551 | 105,212 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 23 | E5 SwiGLU | C3_elem | 99,490 | 99,523 | 33 | ceil(4224/128)=33 |
| 24 | E5 down [3,1408,1024]×2 stream (dual-VC  | C3_VC | 99,523 | 122,184 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 25 | E2 gate+up [4,2048,1408] stream (dual-VC | C2_VC | 105,212 | 150,449 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 26 | E3 gate+up [2,2048,1408] stream (dual-VC | C3_VC | 122,184 | 167,246 | 45,062 | pertile: 88tiles tile0=513 pipe=512 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 27 | E2 SwiGLU | C2_elem | 150,449 | 150,493 | 44 | ceil(5632/128)=44 |
| 28 | E2 down [4,1408,1024]×2 stream (dual-VC  | C2_VC | 150,493 | 173,154 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 29 | E3 SwiGLU | C3_elem | 167,246 | 167,268 | 22 | ceil(2816/128)=22 |
| 30 | E3 down [2,1408,1024]×2 stream (dual-VC  | C3_VC | 167,268 | 189,802 | 22,534 | pertile: 64tiles tile0=353 pipe=352 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 31 | E4 gate+up [4,2048,1408] stream (dual-VC | C2_VC | 173,154 | 218,391 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 32 | C1 SiLU (45056 elem) | C1_elem | 182,149 | 182,501 | 352 | ceil(45056/128)=352 |
| 33 | C1 GLU (45056 elem) | C1_elem | 182,501 | 182,853 | 352 | ceil(45056/128)=352 |
| 34 | active_A C1→C0 (45056B) | DMA_C1_xDMA↔C0_xDMA | 182,853 | 183,557 | 704 | ceil(45056/64)=704 |
| 35 | C1 half_down [16,2816,1024] | C1_VC | 182,853 | 273,482 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 36 | C0 half_down [16,2816,1024] | C0_VC | 182,897 | 273,526 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 37 | E7 gate+up [2,2048,1408] stream (dual-VC | C3_VC | 189,802 | 234,864 | 45,062 | pertile: 88tiles tile0=513 pipe=512 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 38 | E4 SwiGLU | C2_elem | 218,391 | 218,435 | 44 | ceil(5632/128)=44 |
| 39 | E4 down [4,1408,1024]×2 stream (dual-VC  | C2_VC | 218,435 | 241,096 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 40 | E7 SwiGLU | C3_elem | 234,864 | 234,886 | 22 | ceil(2816/128)=22 |
| 41 | E7 down [2,1408,1024]×2 stream (dual-VC  | C3_VC | 234,886 | 257,420 | 22,534 | pertile: 64tiles tile0=353 pipe=352 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 42 | E8 gate+up [3,2048,1408] stream (dual-VC | C2_VC | 241,096 | 286,333 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 43 | E9 gate+up [2,2048,1408] stream (dual-VC | C3_VC | 257,420 | 302,482 | 45,062 | pertile: 88tiles tile0=513 pipe=512 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 44 | merge half_down (16384B) | DMA_C1_xDMA↔C0_xDMA | 273,526 | 273,782 | 256 | ceil(16384/64)=256 |
| 45 | E8 SwiGLU | C2_elem | 286,333 | 286,366 | 33 | ceil(4224/128)=33 |
| 46 | E8 down [3,1408,1024]×2 stream (dual-VC  | C2_VC | 286,366 | 309,027 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 47 | E9 SwiGLU | C3_elem | 302,482 | 302,504 | 22 | ceil(2816/128)=22 |
| 48 | E9 down [2,1408,1024]×2 stream (dual-VC  | C3_VC | 302,504 | 325,038 | 22,534 | pertile: 64tiles tile0=353 pipe=352 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 49 | E10 gate+up [3,2048,1408] stream (dual-V | C2_VC | 309,027 | 354,264 | 45,237 | pertile: 176tiles tile0=257 pipe=257 dma_total=45056 bank_s=1.00 →45237 [compute-bound] |
| 50 | E11 gate+up [2,2048,1408] stream (dual-V | C3_VC | 325,038 | 370,100 | 45,062 | pertile: 88tiles tile0=513 pipe=512 dma_total=45056 bank_s=1.00 →45062 [DMA-bound] |
| 51 | E10 SwiGLU | C2_elem | 354,264 | 354,297 | 33 | ceil(4224/128)=33 |
| 52 | E10 down [3,1408,1024]×2 stream (dual-VC | C2_VC | 354,297 | 376,958 | 22,661 | pertile: 128tiles tile0=177 pipe=177 dma_total=22528 bank_s=1.00 →22661 [compute-bound] |
| 53 | E11 SwiGLU | C3_elem | 370,100 | 370,122 | 22 | ceil(2816/128)=22 |
| 54 | E11 down [2,1408,1024]×2 stream (dual-VC | C3_VC | 370,122 | 392,656 | 22,534 | pertile: 64tiles tile0=353 pipe=352 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |

#### 调度决策表 (M=16, 策略=unified_dynamic)

- Token分布: 12experts: [4, 4, 4, 3, 3, 3, 2, 2, 2, 2]...(2 more)
- Routed CC: 392,656, Shared CC: 273,782, Ratio: 1.434
- VC利用率: 74.6%, xDMA利用率: 91.2%, iDMA利用率: 90.9%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E1 | 2 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 33,976 | 缓存命中 resident 2tok @C3 (省50%) |
| E6 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E0 | 4 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 100% | 67,942 | unified_parallel @64B/cc slack=314 |
| E5 | 3 | C3 | [4x8x8] | idma | 64 | 1 | 否 | 75% | 67,931 | unified_parallel @64B/cc slack=314 |
| E2 | 4 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 100% | 67,942 | unified_parallel @64B/cc slack=314 |
| E3 | 2 | C3 | [2x8x16] | idma | 64 | 1 | 否 | 50% | 67,618 | unified_parallel @64B/cc  |
| E4 | 4 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 100% | 67,942 | unified_parallel @64B/cc slack=314 |
| E7 | 2 | C3 | [2x8x16] | idma | 64 | 1 | 否 | 50% | 67,618 | unified_parallel @64B/cc  |
| E8 | 3 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 75% | 67,931 | unified_parallel @64B/cc slack=314 |
| E9 | 2 | C3 | [2x8x16] | idma | 64 | 1 | 否 | 50% | 67,618 | unified_parallel @64B/cc  |
| E10 | 3 | C2 | [4x8x8] | xdma | 64 | 1 | 否 | 75% | 67,931 | unified_parallel @64B/cc slack=314 |
| E11 | 2 | C3 | [2x8x16] | idma | 64 | 1 | 否 | 50% | 67,618 | unified_parallel @64B/cc  |

---

**M=16 最差案例** (ratio=3.841)

### M=16 任务流表 (dist: 32experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(22 more))

| Start | End | Dur | C0_VC | C0_xDMA↔C1_xDMA | C1_VC | C1_elem | C2_VC | C2_elem | C3_VC | C3_elem | DMA_C1_xDMA↔C0_xDMA | DMA_sram_xDMA↔C0_xDMA | DMA_sram_xDMA↔C3_xDMA | Host | SRAM(iDMA)→C3 | SRAM(xDMA)→C2 | iDMA→C1 | iDMA→C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 512 | 512 |  |  |  |  |  |  |  |  |  | token_A→C0 (32768B) |  |  |  |  | token_A→C1 (32768B) |  |
| 512 | 1,024 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  | token_A sram→C3 (32768B) |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,024 | 1,536 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  |  |  |  |  | router_w iDMA→C3 (65536B) |
| 1,536 | 9,765 | 8,229 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | router [16,2048,64] |  |  |  |  |  |  |  |  |  |
| 9,765 | 14,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | topK |  |  |  |  |
| 14,765 | 19,765 | 5,000 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | scatter |  |  |  |  |
| 19,765 | 20,277 | 512 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  |  |  |  |  | softmax | token_A→C3 (32768B) | token_A→C2 (32768B) |  |  |
| 20,277 | 31,590 | 11,313 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 gate+up [1,2048,1408] resid |  | E1 gate+up [1,2048,1408] resid |  |  |  |  | softmax |  |  |  |  |
| 31,590 | 31,601 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E0 SwiGLU |  | E1 SwiGLU |  |  |  | softmax |  |  |  |  |
| 31,601 | 34,765 | 3,164 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  | softmax |  |  |  |  |
| 34,765 | 37,270 | 2,505 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E0 down [1,1408,1024]×2 reside |  | E1 down [1,1408,1024]×2 reside |  |  |  |  |  |  |  |  |  |
| 37,270 | 59,804 | 22,534 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 59,804 | 59,815 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E2 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 59,815 | 71,079 | 11,264 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 71,079 | 71,085 | 6 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E2 down [1,1408,1024]×2 stream |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 71,085 | 93,613 | 22,528 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | E3 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 93,613 | 93,624 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  | E3 SwiGLU |  |  |  |  |  |  |  |  |
| 93,624 | 104,888 | 11,264 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 104,888 | 104,894 | 6 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E4 gate+up [1,2048,1408] strea |  | E3 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 104,894 | 127,422 | 22,528 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E4 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 127,422 | 127,433 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  | E4 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 127,433 | 138,697 | 11,264 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E4 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 138,697 | 138,703 | 6 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E4 down [1,1408,1024]×2 stream |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 138,703 | 161,231 | 22,528 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | E5 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 161,231 | 161,242 | 11 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  |  | E5 SwiGLU |  |  |  |  |  |  |  |  |
| 161,242 | 172,506 | 11,264 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  |  |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 172,506 | 172,512 | 6 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E6 gate+up [1,2048,1408] strea |  | E5 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 172,512 | 182,149 | 9,637 | C0 up_proj [16,2048,2816] | up_result P2P C0→C1 (pipeline) | C1 gate_proj [16,2048,2816] |  | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 182,149 | 182,193 | 44 |  | up_result P2P C0→C1 (pipeline) |  | C1 SiLU (45056 elem) | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 182,193 | 182,501 | 308 |  |  |  | C1 SiLU (45056 elem) | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 182,501 | 182,853 | 352 |  |  |  | C1 GLU (45056 elem) | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 182,853 | 182,897 | 44 |  |  | C1 half_down [16,2816,1024] |  | E6 gate+up [1,2048,1408] strea |  |  |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 182,897 | 183,557 | 660 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E6 gate+up [1,2048,1408] strea |  |  |  | active_A C1→C0 (45056B) |  |  |  |  |  |  |  |
| 183,557 | 195,040 | 11,483 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E6 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 195,040 | 195,051 | 11 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  | E6 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 195,051 | 206,315 | 11,264 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E6 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 206,315 | 206,321 | 6 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E6 down [1,1408,1024]×2 stream |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 206,321 | 228,849 | 22,528 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  |  | E7 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 228,849 | 228,860 | 11 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  |  |  | E7 SwiGLU |  |  |  |  |  |  |  |  |
| 228,860 | 240,124 | 11,264 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  |  | E7 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 240,124 | 240,130 | 6 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E8 gate+up [1,2048,1408] strea |  | E7 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 240,130 | 262,658 | 22,528 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E8 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |  |  |
| 262,658 | 262,669 | 11 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  |  | E8 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 262,669 | 273,482 | 10,813 | C0 half_down [16,2816,1024] |  | C1 half_down [16,2816,1024] |  | E8 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 273,482 | 273,526 | 44 | C0 half_down [16,2816,1024] |  |  |  | E8 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 273,526 | 273,782 | 256 |  |  |  |  | E8 down [1,1408,1024]×2 stream |  |  |  | merge half_down (16384B) |  |  |  |  |  |  |  |
| 273,782 | 273,933 | 151 |  |  |  |  | E8 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |  |  |
| 273,933 | 273,939 | 6 |  |  |  |  | E8 down [1,1408,1024]×2 stream |  | E9 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 273,939 | 296,467 | 22,528 |  |  |  |  |  |  | E9 gate+up [1,2048,1408] strea |  |  |  |  |  |  |  |  |  |
| 296,467 | 296,478 | 11 |  |  |  |  |  |  |  | E9 SwiGLU |  |  |  |  |  |  |  |  |
| 296,478 | 307,742 | 11,264 |  |  |  |  |  |  | E9 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 307,742 | 307,748 | 6 |  |  |  |  | E10 gate+up [1,2048,1408] stre |  | E9 down [1,1408,1024]×2 stream |  |  |  |  |  |  |  |  |  |
| 307,748 | 330,276 | 22,528 |  |  |  |  | E10 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 330,276 | 330,287 | 11 |  |  |  |  |  | E10 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 330,287 | 341,551 | 11,264 |  |  |  |  | E10 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 341,551 | 341,557 | 6 |  |  |  |  | E10 down [1,1408,1024]×2 strea |  | E11 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 341,557 | 364,085 | 22,528 |  |  |  |  |  |  | E11 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 364,085 | 364,096 | 11 |  |  |  |  |  |  |  | E11 SwiGLU |  |  |  |  |  |  |  |  |
| 364,096 | 375,360 | 11,264 |  |  |  |  |  |  | E11 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 375,360 | 375,366 | 6 |  |  |  |  | E12 gate+up [1,2048,1408] stre |  | E11 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 375,366 | 397,894 | 22,528 |  |  |  |  | E12 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 397,894 | 397,905 | 11 |  |  |  |  |  | E12 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 397,905 | 409,169 | 11,264 |  |  |  |  | E12 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 409,169 | 409,175 | 6 |  |  |  |  | E12 down [1,1408,1024]×2 strea |  | E13 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 409,175 | 431,703 | 22,528 |  |  |  |  |  |  | E13 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 431,703 | 431,714 | 11 |  |  |  |  |  |  |  | E13 SwiGLU |  |  |  |  |  |  |  |  |
| 431,714 | 442,978 | 11,264 |  |  |  |  |  |  | E13 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 442,978 | 442,984 | 6 |  |  |  |  | E14 gate+up [1,2048,1408] stre |  | E13 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 442,984 | 465,512 | 22,528 |  |  |  |  | E14 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 465,512 | 465,523 | 11 |  |  |  |  |  | E14 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 465,523 | 476,787 | 11,264 |  |  |  |  | E14 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 476,787 | 476,793 | 6 |  |  |  |  | E14 down [1,1408,1024]×2 strea |  | E15 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 476,793 | 499,321 | 22,528 |  |  |  |  |  |  | E15 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 499,321 | 499,332 | 11 |  |  |  |  |  |  |  | E15 SwiGLU |  |  |  |  |  |  |  |  |
| 499,332 | 510,596 | 11,264 |  |  |  |  |  |  | E15 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 510,596 | 510,602 | 6 |  |  |  |  | E16 gate+up [1,2048,1408] stre |  | E15 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 510,602 | 533,130 | 22,528 |  |  |  |  | E16 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 533,130 | 533,141 | 11 |  |  |  |  |  | E16 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 533,141 | 544,405 | 11,264 |  |  |  |  | E16 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 544,405 | 544,411 | 6 |  |  |  |  | E16 down [1,1408,1024]×2 strea |  | E17 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 544,411 | 566,939 | 22,528 |  |  |  |  |  |  | E17 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 566,939 | 566,950 | 11 |  |  |  |  |  |  |  | E17 SwiGLU |  |  |  |  |  |  |  |  |
| 566,950 | 578,214 | 11,264 |  |  |  |  |  |  | E17 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 578,214 | 578,220 | 6 |  |  |  |  | E18 gate+up [1,2048,1408] stre |  | E17 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 578,220 | 600,748 | 22,528 |  |  |  |  | E18 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 600,748 | 600,759 | 11 |  |  |  |  |  | E18 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 600,759 | 612,023 | 11,264 |  |  |  |  | E18 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 612,023 | 612,029 | 6 |  |  |  |  | E18 down [1,1408,1024]×2 strea |  | E19 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 612,029 | 634,557 | 22,528 |  |  |  |  |  |  | E19 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 634,557 | 634,568 | 11 |  |  |  |  |  |  |  | E19 SwiGLU |  |  |  |  |  |  |  |  |
| 634,568 | 645,832 | 11,264 |  |  |  |  |  |  | E19 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 645,832 | 645,838 | 6 |  |  |  |  | E20 gate+up [1,2048,1408] stre |  | E19 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 645,838 | 668,366 | 22,528 |  |  |  |  | E20 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 668,366 | 668,377 | 11 |  |  |  |  |  | E20 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 668,377 | 679,641 | 11,264 |  |  |  |  | E20 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 679,641 | 679,647 | 6 |  |  |  |  | E20 down [1,1408,1024]×2 strea |  | E21 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 679,647 | 702,175 | 22,528 |  |  |  |  |  |  | E21 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 702,175 | 702,186 | 11 |  |  |  |  |  |  |  | E21 SwiGLU |  |  |  |  |  |  |  |  |
| 702,186 | 713,450 | 11,264 |  |  |  |  |  |  | E21 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 713,450 | 713,456 | 6 |  |  |  |  | E22 gate+up [1,2048,1408] stre |  | E21 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 713,456 | 735,984 | 22,528 |  |  |  |  | E22 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 735,984 | 735,995 | 11 |  |  |  |  |  | E22 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 735,995 | 747,259 | 11,264 |  |  |  |  | E22 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 747,259 | 747,265 | 6 |  |  |  |  | E22 down [1,1408,1024]×2 strea |  | E23 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 747,265 | 769,793 | 22,528 |  |  |  |  |  |  | E23 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 769,793 | 769,804 | 11 |  |  |  |  |  |  |  | E23 SwiGLU |  |  |  |  |  |  |  |  |
| 769,804 | 781,068 | 11,264 |  |  |  |  |  |  | E23 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 781,068 | 781,074 | 6 |  |  |  |  | E24 gate+up [1,2048,1408] stre |  | E23 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 781,074 | 803,602 | 22,528 |  |  |  |  | E24 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 803,602 | 803,613 | 11 |  |  |  |  |  | E24 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 803,613 | 814,877 | 11,264 |  |  |  |  | E24 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 814,877 | 814,883 | 6 |  |  |  |  | E24 down [1,1408,1024]×2 strea |  | E25 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 814,883 | 837,411 | 22,528 |  |  |  |  |  |  | E25 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 837,411 | 837,422 | 11 |  |  |  |  |  |  |  | E25 SwiGLU |  |  |  |  |  |  |  |  |
| 837,422 | 848,686 | 11,264 |  |  |  |  |  |  | E25 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 848,686 | 848,692 | 6 |  |  |  |  | E26 gate+up [1,2048,1408] stre |  | E25 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 848,692 | 871,220 | 22,528 |  |  |  |  | E26 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 871,220 | 871,231 | 11 |  |  |  |  |  | E26 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 871,231 | 882,495 | 11,264 |  |  |  |  | E26 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 882,495 | 882,501 | 6 |  |  |  |  | E26 down [1,1408,1024]×2 strea |  | E27 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 882,501 | 905,029 | 22,528 |  |  |  |  |  |  | E27 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 905,029 | 905,040 | 11 |  |  |  |  |  |  |  | E27 SwiGLU |  |  |  |  |  |  |  |  |
| 905,040 | 916,304 | 11,264 |  |  |  |  |  |  | E27 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 916,304 | 916,310 | 6 |  |  |  |  | E28 gate+up [1,2048,1408] stre |  | E27 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 916,310 | 938,838 | 22,528 |  |  |  |  | E28 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 938,838 | 938,849 | 11 |  |  |  |  |  | E28 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 938,849 | 950,113 | 11,264 |  |  |  |  | E28 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 950,113 | 950,119 | 6 |  |  |  |  | E28 down [1,1408,1024]×2 strea |  | E29 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 950,119 | 972,647 | 22,528 |  |  |  |  |  |  | E29 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 972,647 | 972,658 | 11 |  |  |  |  |  |  |  | E29 SwiGLU |  |  |  |  |  |  |  |  |
| 972,658 | 983,922 | 11,264 |  |  |  |  |  |  | E29 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 983,922 | 983,928 | 6 |  |  |  |  | E30 gate+up [1,2048,1408] stre |  | E29 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |
| 983,928 | 1,006,456 | 22,528 |  |  |  |  | E30 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |  |  |
| 1,006,456 | 1,006,467 | 11 |  |  |  |  |  | E30 SwiGLU |  |  |  |  |  |  |  |  |  |  |
| 1,006,467 | 1,017,731 | 11,264 |  |  |  |  | E30 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |  |  |
| 1,017,731 | 1,017,737 | 6 |  |  |  |  | E30 down [1,1408,1024]×2 strea |  | E31 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 1,017,737 | 1,040,265 | 22,528 |  |  |  |  |  |  | E31 gate+up [1,2048,1408] stre |  |  |  |  |  |  |  |  |  |
| 1,040,265 | 1,040,276 | 11 |  |  |  |  |  |  |  | E31 SwiGLU |  |  |  |  |  |  |  |  |
| 1,040,276 | 1,051,546 | 11,270 |  |  |  |  |  |  | E31 down [1,1408,1024]×2 strea |  |  |  |  |  |  |  |  |  |

#### TCDM状态 (M=16)

| 时刻 | Cluster | 内容 | 已用 | 剩余 |
|------|---------|------|------|------|
| 273,782 | C0 | up_weight:2.750MB, half_down_first:1.375MB | 4.125MB | 0.875MB |
| 273,782 | C1 | gate_weight:2.750MB, half_down_second:1.375MB | 4.125MB | 0.875MB |
| 71,085 | C2 | E2_weights:4.125MB | 4.125MB | 0.875MB |
| 104,894 | C3 | E3_weights:4.125MB | 4.125MB | 0.875MB |
| 138,703 | C2 | E4_weights:4.125MB | 4.125MB | 0.875MB |
| 172,512 | C3 | E5_weights:4.125MB | 4.125MB | 0.875MB |
| 206,321 | C2 | E6_weights:4.125MB | 4.125MB | 0.875MB |
| 240,130 | C3 | E7_weights:4.125MB | 4.125MB | 0.875MB |
| 273,939 | C2 | E8_weights:4.125MB | 4.125MB | 0.875MB |
| 307,748 | C3 | E9_weights:4.125MB | 4.125MB | 0.875MB |
| 341,557 | C2 | E10_weights:4.125MB | 4.125MB | 0.875MB |
| 375,366 | C3 | E11_weights:4.125MB | 4.125MB | 0.875MB |
| 409,175 | C2 | E12_weights:4.125MB | 4.125MB | 0.875MB |
| 442,984 | C3 | E13_weights:4.125MB | 4.125MB | 0.875MB |
| 476,793 | C2 | E14_weights:4.125MB | 4.125MB | 0.875MB |
| 510,602 | C3 | E15_weights:4.125MB | 4.125MB | 0.875MB |
| 544,411 | C2 | E16_weights:4.125MB | 4.125MB | 0.875MB |
| 578,220 | C3 | E17_weights:4.125MB | 4.125MB | 0.875MB |
| 612,029 | C2 | E18_weights:4.125MB | 4.125MB | 0.875MB |
| 645,838 | C3 | E19_weights:4.125MB | 4.125MB | 0.875MB |
| 679,647 | C2 | E20_weights:4.125MB | 4.125MB | 0.875MB |
| 713,456 | C3 | E21_weights:4.125MB | 4.125MB | 0.875MB |
| 747,265 | C2 | E22_weights:4.125MB | 4.125MB | 0.875MB |
| 781,074 | C3 | E23_weights:4.125MB | 4.125MB | 0.875MB |
| 814,883 | C2 | E24_weights:4.125MB | 4.125MB | 0.875MB |
| 848,692 | C3 | E25_weights:4.125MB | 4.125MB | 0.875MB |
| 882,501 | C2 | E26_weights:4.125MB | 4.125MB | 0.875MB |
| 916,310 | C3 | E27_weights:4.125MB | 4.125MB | 0.875MB |
| 950,119 | C2 | E28_weights:4.125MB | 4.125MB | 0.875MB |
| 983,928 | C3 | E29_weights:4.125MB | 4.125MB | 0.875MB |
| 1,017,737 | C2 | E30_weights:4.125MB | 4.125MB | 0.875MB |
| 1,051,546 | C3 | E31_weights:4.125MB | 4.125MB | 0.875MB |

#### 持续时间公式表 (M=16)

| # | Task | Resource | Start | End | Duration | Formula |
|---|------|----------|-------|-----|----------|---------|
| 0 | token_A→C0 (32768B) | DMA_sram_xDMA↔C0_xDMA | 0 | 512 | 512 | ceil(32768/64)=512 |
| 1 | token_A→C1 (32768B) | iDMA→C1 | 0 | 512 | 512 | ceil(32768/64)=512 |
| 2 | C0 up_proj [16,2048,2816] | C0_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 3 | C1 gate_proj [16,2048,2816] | C1_VC | 512 | 182,149 | 181,637 | gemm(16,2048,2816,[1x8x64])=181637 util=100% |
| 4 | up_result P2P C0→C1 (pipeline) | C0_xDMA↔C1_xDMA | 512 | 182,193 | 181,681 | pipeline with up_proj, last_row=44cc |
| 5 | router_w iDMA→C3 (65536B) | iDMA→C3 | 512 | 1,536 | 1,024 | ceil(65536/64)=1024 |
| 6 | token_A sram→C3 (32768B) | DMA_sram_xDMA↔C3_xDMA | 512 | 1,024 | 512 | ceil(32768/64)=512 |
| 7 | router [16,2048,64] | C3_VC | 1,536 | 9,765 | 8,229 | gemm(16,2048,64,[2x8x16])=8229 util=100% |
| 8 | topK | Host | 9,765 | 14,765 | 5,000 | ~5000cc overhead |
| 9 | scatter | Host | 14,765 | 19,765 | 5,000 | ~5000cc |
| 10 | softmax | Host | 19,765 | 34,765 | 15,000 | ~15000cc |
| 11 | token_A→C2 (32768B) | SRAM(xDMA)→C2 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [xDMA] |
| 12 | token_A→C3 (32768B) | SRAM(iDMA)→C3 | 19,765 | 20,277 | 512 | ceil(32768/64)=512 [iDMA] |
| 13 | E0 gate+up [1,2048,1408] resident | C2_VC | 20,277 | 31,590 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 14 | E1 gate+up [1,2048,1408] resident | C3_VC | 20,277 | 31,590 | 11,313 | dual_vc_gu_resident: gemm(1,2048,1408,[1x8x32])=11313 util=100% |
| 15 | E0 SwiGLU | C2_elem | 31,590 | 31,601 | 11 | ceil(1408/128)=11 |
| 16 | E1 SwiGLU | C3_elem | 31,590 | 31,601 | 11 | ceil(1408/128)=11 |
| 17 | E0 down [1,1408,1024]×2 resident | C2_VC | 31,601 | 37,270 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 18 | E1 down [1,1408,1024]×2 resident | C3_VC | 31,601 | 37,270 | 5,669 | dual_vc_dn_resident: gemm(1,1408,1024,[1x8x32])=5669 util=100% |
| 19 | E2 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 37,270 | 59,804 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 20 | E2 SwiGLU | C2_elem | 59,804 | 59,815 | 11 | ceil(1408/128)=11 |
| 21 | E2 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 59,815 | 71,085 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 22 | E3 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 71,079 | 93,613 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 23 | E3 SwiGLU | C3_elem | 93,613 | 93,624 | 11 | ceil(1408/128)=11 |
| 24 | E3 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 93,624 | 104,894 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 25 | E4 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 104,888 | 127,422 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 26 | E4 SwiGLU | C2_elem | 127,422 | 127,433 | 11 | ceil(1408/128)=11 |
| 27 | E4 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 127,433 | 138,703 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 28 | E5 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 138,697 | 161,231 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 29 | E5 SwiGLU | C3_elem | 161,231 | 161,242 | 11 | ceil(1408/128)=11 |
| 30 | E5 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 161,242 | 172,512 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 31 | E6 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 172,506 | 195,040 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 32 | C1 SiLU (45056 elem) | C1_elem | 182,149 | 182,501 | 352 | ceil(45056/128)=352 |
| 33 | C1 GLU (45056 elem) | C1_elem | 182,501 | 182,853 | 352 | ceil(45056/128)=352 |
| 34 | active_A C1→C0 (45056B) | DMA_C1_xDMA↔C0_xDMA | 182,853 | 183,557 | 704 | ceil(45056/64)=704 |
| 35 | C1 half_down [16,2816,1024] | C1_VC | 182,853 | 273,482 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 36 | C0 half_down [16,2816,1024] | C0_VC | 182,897 | 273,526 | 90,629 | gemm(16,2816,1024,[1x8x64])=90629 util=100% |
| 37 | E6 SwiGLU | C2_elem | 195,040 | 195,051 | 11 | ceil(1408/128)=11 |
| 38 | E6 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 195,051 | 206,321 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 39 | E7 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 206,315 | 228,849 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 40 | E7 SwiGLU | C3_elem | 228,849 | 228,860 | 11 | ceil(1408/128)=11 |
| 41 | E7 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 228,860 | 240,130 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 42 | E8 gate+up [1,2048,1408] stream (dual-VC | C2_VC | 240,124 | 262,658 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 43 | E8 SwiGLU | C2_elem | 262,658 | 262,669 | 11 | ceil(1408/128)=11 |
| 44 | E8 down [1,1408,1024]×2 stream (dual-VC  | C2_VC | 262,669 | 273,939 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 45 | merge half_down (16384B) | DMA_C1_xDMA↔C0_xDMA | 273,526 | 273,782 | 256 | ceil(16384/64)=256 |
| 46 | E9 gate+up [1,2048,1408] stream (dual-VC | C3_VC | 273,933 | 296,467 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 47 | E9 SwiGLU | C3_elem | 296,467 | 296,478 | 11 | ceil(1408/128)=11 |
| 48 | E9 down [1,1408,1024]×2 stream (dual-VC  | C3_VC | 296,478 | 307,748 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 49 | E10 gate+up [1,2048,1408] stream (dual-V | C2_VC | 307,742 | 330,276 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 50 | E10 SwiGLU | C2_elem | 330,276 | 330,287 | 11 | ceil(1408/128)=11 |
| 51 | E10 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 330,287 | 341,557 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 52 | E11 gate+up [1,2048,1408] stream (dual-V | C3_VC | 341,551 | 364,085 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 53 | E11 SwiGLU | C3_elem | 364,085 | 364,096 | 11 | ceil(1408/128)=11 |
| 54 | E11 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 364,096 | 375,366 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 55 | E12 gate+up [1,2048,1408] stream (dual-V | C2_VC | 375,360 | 397,894 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 56 | E12 SwiGLU | C2_elem | 397,894 | 397,905 | 11 | ceil(1408/128)=11 |
| 57 | E12 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 397,905 | 409,175 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 58 | E13 gate+up [1,2048,1408] stream (dual-V | C3_VC | 409,169 | 431,703 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 59 | E13 SwiGLU | C3_elem | 431,703 | 431,714 | 11 | ceil(1408/128)=11 |
| 60 | E13 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 431,714 | 442,984 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 61 | E14 gate+up [1,2048,1408] stream (dual-V | C2_VC | 442,978 | 465,512 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 62 | E14 SwiGLU | C2_elem | 465,512 | 465,523 | 11 | ceil(1408/128)=11 |
| 63 | E14 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 465,523 | 476,793 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 64 | E15 gate+up [1,2048,1408] stream (dual-V | C3_VC | 476,787 | 499,321 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 65 | E15 SwiGLU | C3_elem | 499,321 | 499,332 | 11 | ceil(1408/128)=11 |
| 66 | E15 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 499,332 | 510,602 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 67 | E16 gate+up [1,2048,1408] stream (dual-V | C2_VC | 510,596 | 533,130 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 68 | E16 SwiGLU | C2_elem | 533,130 | 533,141 | 11 | ceil(1408/128)=11 |
| 69 | E16 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 533,141 | 544,411 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 70 | E17 gate+up [1,2048,1408] stream (dual-V | C3_VC | 544,405 | 566,939 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 71 | E17 SwiGLU | C3_elem | 566,939 | 566,950 | 11 | ceil(1408/128)=11 |
| 72 | E17 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 566,950 | 578,220 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 73 | E18 gate+up [1,2048,1408] stream (dual-V | C2_VC | 578,214 | 600,748 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 74 | E18 SwiGLU | C2_elem | 600,748 | 600,759 | 11 | ceil(1408/128)=11 |
| 75 | E18 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 600,759 | 612,029 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 76 | E19 gate+up [1,2048,1408] stream (dual-V | C3_VC | 612,023 | 634,557 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 77 | E19 SwiGLU | C3_elem | 634,557 | 634,568 | 11 | ceil(1408/128)=11 |
| 78 | E19 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 634,568 | 645,838 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 79 | E20 gate+up [1,2048,1408] stream (dual-V | C2_VC | 645,832 | 668,366 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 80 | E20 SwiGLU | C2_elem | 668,366 | 668,377 | 11 | ceil(1408/128)=11 |
| 81 | E20 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 668,377 | 679,647 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 82 | E21 gate+up [1,2048,1408] stream (dual-V | C3_VC | 679,641 | 702,175 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 83 | E21 SwiGLU | C3_elem | 702,175 | 702,186 | 11 | ceil(1408/128)=11 |
| 84 | E21 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 702,186 | 713,456 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 85 | E22 gate+up [1,2048,1408] stream (dual-V | C2_VC | 713,450 | 735,984 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 86 | E22 SwiGLU | C2_elem | 735,984 | 735,995 | 11 | ceil(1408/128)=11 |
| 87 | E22 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 735,995 | 747,265 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 88 | E23 gate+up [1,2048,1408] stream (dual-V | C3_VC | 747,259 | 769,793 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 89 | E23 SwiGLU | C3_elem | 769,793 | 769,804 | 11 | ceil(1408/128)=11 |
| 90 | E23 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 769,804 | 781,074 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 91 | E24 gate+up [1,2048,1408] stream (dual-V | C2_VC | 781,068 | 803,602 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 92 | E24 SwiGLU | C2_elem | 803,602 | 803,613 | 11 | ceil(1408/128)=11 |
| 93 | E24 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 803,613 | 814,883 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 94 | E25 gate+up [1,2048,1408] stream (dual-V | C3_VC | 814,877 | 837,411 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 95 | E25 SwiGLU | C3_elem | 837,411 | 837,422 | 11 | ceil(1408/128)=11 |
| 96 | E25 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 837,422 | 848,692 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 97 | E26 gate+up [1,2048,1408] stream (dual-V | C2_VC | 848,686 | 871,220 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 98 | E26 SwiGLU | C2_elem | 871,220 | 871,231 | 11 | ceil(1408/128)=11 |
| 99 | E26 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 871,231 | 882,501 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 100 | E27 gate+up [1,2048,1408] stream (dual-V | C3_VC | 882,495 | 905,029 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 101 | E27 SwiGLU | C3_elem | 905,029 | 905,040 | 11 | ceil(1408/128)=11 |
| 102 | E27 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 905,040 | 916,310 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 103 | E28 gate+up [1,2048,1408] stream (dual-V | C2_VC | 916,304 | 938,838 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 104 | E28 SwiGLU | C2_elem | 938,838 | 938,849 | 11 | ceil(1408/128)=11 |
| 105 | E28 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 938,849 | 950,119 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 106 | E29 gate+up [1,2048,1408] stream (dual-V | C3_VC | 950,113 | 972,647 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 107 | E29 SwiGLU | C3_elem | 972,647 | 972,658 | 11 | ceil(1408/128)=11 |
| 108 | E29 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 972,658 | 983,928 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 109 | E30 gate+up [1,2048,1408] stream (dual-V | C2_VC | 983,922 | 1,006,456 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 110 | E30 SwiGLU | C2_elem | 1,006,456 | 1,006,467 | 11 | ceil(1408/128)=11 |
| 111 | E30 down [1,1408,1024]×2 stream (dual-VC | C2_VC | 1,006,467 | 1,017,737 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |
| 112 | E31 gate+up [1,2048,1408] stream (dual-V | C3_VC | 1,017,731 | 1,040,265 | 22,534 | pertile: 44tiles tile0=513 pipe=512 dma_total=22528 bank_s=1.00 →22534 [DMA-bound] |
| 113 | E31 SwiGLU | C3_elem | 1,040,265 | 1,040,276 | 11 | ceil(1408/128)=11 |
| 114 | E31 down [1,1408,1024]×2 stream (dual-VC | C3_VC | 1,040,276 | 1,051,546 | 11,270 | pertile: 32tiles tile0=353 pipe=352 dma_total=11264 bank_s=1.00 →11270 [DMA-bound] |

#### 调度决策表 (M=16, 策略=sequential_full)

- Token分布: 32experts: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...(22 more)
- Routed CC: 1,051,546, Shared CC: 273,782, Ratio: 3.841
- VC利用率: 51.9%, xDMA利用率: 100.0%, iDMA利用率: 100.0%

| Expert | Tokens | Cluster | Shape | DMA | BW | Phase | Resident | VC利用率 | Est.CC | 决策理由 |
|--------|--------|---------|-------|-----|-----|-------|----------|---------|--------|---------|
| E0 | 1 | C2 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C2 (省75%) |
| E1 | 1 | C3 | [1x8x32] | none | 0 | 0 | 是 | 100% | 16,993 | 缓存命中 resident 1tok @C3 (省75%) |
| E2 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E3 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E4 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E5 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E6 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E7 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E8 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E9 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E10 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E11 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E12 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E13 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E14 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E15 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E16 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E17 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E18 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E19 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E20 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E21 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E22 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E23 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E24 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E25 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E26 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E27 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E28 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E29 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E30 | 1 | C2 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |
| E31 | 1 | C3 | [1x8x32] | both | 128 | 0 | 否 | 50% | 33,815 | sequential full @128B/cc |

## 7. 分析与结论

### 7.1 DMA带宽瓶颈分析

- 单个routed expert权重: gate+up+down = 3×2048×1408×0.5 = 4.125MB
- @64B/cc搬运时间: 67,584cc (一对多并行搬运时)
- @128B/cc搬运时间: 33,792cc (独享xDMA+iDMA时)

### 7.2 Per-Shape双VC带宽需求分析

| Shape [R×T×C] | 单VC B需求 | 双VC B需求 | K-tile(T cc) | DMA/tile@64B | DMA/tile@128B | @64B stall | @128B stall |
|------|------|------|------|------|------|------|------|
| [1x8x32] | 128B | 256B | 8cc | 4cc | 2cc | NO | NO |
| [2x8x16] | 64B | 128B | 8cc | 2cc | 1cc | NO | NO |
| [4x8x8] | 32B | 64B | 8cc | 1cc | 1cc | NO | NO |
| [8x8x4] | 16B | 32B | 8cc | 1cc | 1cc | NO | NO |
| [16x8x2] | 8B | 16B | 8cc | 1cc | 1cc | NO | NO |
| [32x8x1] | 4B | 8B | 8cc | 1cc | 1cc | NO | NO |

> 注: 双VC B需求 = 2 × T × C × wpe (两个VC各自独立读A和B)
> Bank需求 = 2×A_banks + 2×B_banks (无broadcast)
> [4×8×8]双VC B=64B/cc, bank=16, 恰好匹配单通道DMA(64B/cc)
> [2×8×16]双VC B=128B/cc, bank=20, 需要xDMA+iDMA同时工作(128B/cc)
> [1×8×32]双VC B=256B/cc, bank=34, 即使128B/cc也不够, DMA-bound不可避免

### 7.3 Routed vs Shared 时间比分析 (Dual-VC模型)

并行调度下(C2=xDMA@64B/cc, C3=iDMA@64B/cc), 单expert时间:

| M | gu_compute | dn_compute | gu_dma | dn_dma | stream_total | shared_cc | Ratio理论 |
|---|-----------|-----------|--------|--------|-------------|----------|----------|
| 1 | 45,237 | 22,661 | 45,056 | 22,528 | 68,213 | 17,184 | 3.970 |
| 4 | 45,237 | 22,661 | 45,056 | 22,528 | 68,246 | 68,486 | 0.996 |
| 8 | 90,469 | 45,317 | 45,056 | 22,528 | 136,178 | 136,918 | 0.995 |
| 16 | 180,933 | 90,629 | 45,056 | 22,528 | 272,042 | 273,782 | 0.994 |
| 64 | 723,717 | 362,501 | 45,056 | 22,528 | 1,087,226 | 1,094,966 | 0.993 |
| 128 | 1,447,429 | 724,997 | 45,056 | 22,528 | 2,174,138 | 2,189,878 | 0.993 |

> 使用[4×8×8]@64B/cc, 双VC B需求=64B/cc, 恰好匹配单通道DMA。
> M=1时 gu_compute≈gu_dma → 刚好平衡。
> M≥4时 compute远超DMA → compute-bound, DMA完全overlap。
> 两个expert并行后, 理论ratio≈stream_total/shared_cc。

### 7.4 关键发现

- **M=1**: shared=17,184cc, NC ratio=[1.977, 3.936], NC avg=2.956, C avg=1.483 (-49.8%), NC≤1.1: 0.0%, C≤1.1: 50.0%
- **M=2**: shared=34,270cc, NC ratio=[1.982, 3.947], NC avg=2.572, C avg=1.684 (-34.5%), NC≤1.1: 0.0%, C≤1.1: 20.0%
- **M=4**: shared=68,486cc, NC ratio=[0.992, 3.950], NC avg=2.096, C avg=1.588 (-24.2%), NC≤1.1: 11.8%, C≤1.1: 23.5%
- **M=8**: shared=136,918cc, NC ratio=[0.992, 3.952], NC avg=2.020, C avg=1.719 (-14.9%), NC≤1.1: 7.0%, C≤1.1: 9.3%
- **M=16**: shared=273,782cc, NC ratio=[0.993, 3.952], NC avg=1.719, C avg=1.585 (-7.8%), NC≤1.1: 5.0%, C≤1.1: 14.5%
- **M=64**: shared=1,094,966cc, NC ratio=[0.993, 2.287], NC avg=1.410, C avg=1.377 (-2.3%), NC≤1.1: 19.5%, C≤1.1: 25.5%
- **M=128**: shared=2,189,878cc, NC ratio=[0.993, 1.582], NC avg=1.174, C avg=1.163 (-0.9%), NC≤1.1: 48.0%, C≤1.1: 48.0%

### 7.5 调度策略效果分析

- **phase_based**: 对token集中的分布(2-4 active experts)最有效, 可以充分利用xDMA+iDMA并行, ratio接近1.0
- **greedy_balanced**: 对token分散的分布(多个cold expert)较好, 负载均衡减少最长路径
- **sequential_full**: 仅在极端情况(1个expert独占)有优势, 独享128B/cc带宽
- **bw_steal**: 利用先完成的cluster释放的DMA通道, 特别适合一热一冷组合场景
- **adaptive_split**: 穷举热门expert的拆分点, 将一个热门拆分到两个cluster并行, 适合1热多冷
- **online_greedy**: 真正的在线动态调度, 每步视野最优, 对复杂分布(多种token数混合)效果最好
- **cold_batch**: 先并行@64处理热门, 然后批量@128消化冷门, 适合冷门expert特别多的分布
- **unified_dynamic**: 融合所有策略 + DMA预取 + 专家克隆, 极端场景(1-2 expert超多token)最优
- **event_driven**: DMA/计算解耦, M≥8的通用最优策略, v18核心创新

### 7.6 缓存效果分析

- **小M (M=1~2)**: 缓存效果最好, ntok=1时节省75%, 平均ratio改善35-50%
- **中M (M=4~16)**: 缓存仍有显著效果, ratio改善8-24%, 缓存旁路机制避免大token的负载不均
- **大M (M=64~128)**: 缓存效果有限(<3%), 因为active experts多且token数大, compute-bound为主
- **自适应旁路**: 当ntok≥4时自动跳过缓存, 避免了phase_based在M≥64时的+3~5%性能退化

### 7.7 DMA瓶颈的不可避免性

- **核心限制**: 当active expert数 >> 2时, DMA带宽是不可避免的瓶颈。
  每对expert需要~67,584cc DMA时间, 而shared expert的计算也只有~M×17,000cc。
  n_pair = ceil(n_active/2), 串行DMA总时间 = n_pair × 67,584cc。
- 缓存最多减少2个expert的DMA时间, 对大量active experts场景帮助有限。

## 8. C语言部署参考

```c
// 调度决策逻辑 (伪代码)
void moe_schedule(int M, int* topk_results, int* token_counts,
                  int c2_cached_eid, int c3_cached_eid) {
    // 1. 检查缓存命中
    for (int i = 0; i < n_active_experts; i++) {
        int eid = topk_results[i];
        int ntok = token_counts[i];
        if (eid == c2_cached_eid && ntok <= 2) {
            // 缓存命中且有收益: resident模式 @C2
            schedule_resident(eid, ntok, C2);
        } else if (eid == c3_cached_eid && ntok <= 2) {
            schedule_resident(eid, ntok, C3);
        } else {
            // 正常DMA调度
            schedule_streaming(eid, ntok, best_strategy_from_LUT(M));
        }
    }
    // 2. 更新exit_eids
    c2_cached_eid = last_expert_on_C2;
    c3_cached_eid = last_expert_on_C3;
}
```

**关键参数**:
- 缓存容量: 每cluster 1个完整expert (4.125MB / 5MB TCDM)
- DMA带宽: sram_xDMA=64B/cc, iDMA=64B/cc
- 缓存旁路阈值: ntok≥4时跳过 (节省<5%)
- exit_eids更新: 每个cluster最后执行的expert → 下一层缓存映射
