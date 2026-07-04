# MoE 分析调度算法 — 逐函数详解

> 对应源文件：`Idea_Model/c_interface/moe_scheduler.c`  
> Python 参考：`Idea_Model/analytical_scheduler.py`

---

## 0. 系统背景

### 硬件拓扑

```
Host CVA6
  └── C2 cluster  (iDMA / xDMA 控制，L1 TCDM)
  └── C3 cluster  (iDMA / xDMA 控制，L1 TCDM)
共享 HBM ─── 最大聚合带宽 MAX_BW = 128 B/cc
```

调度器的任务是：给定一批 expert（ID + token 数），决定每个 expert 在哪个 cluster 上跑、用什么矩阵分块规格（shape）、何时启动 DMA、是否预取——使总 makespan 最短。

---

## 1. 物理常数与 Shape 定义

### 1.1 物理常数

| 符号 | 值 | 含义 |
|---|---|---|
| `WBYTES_S1` | 2,883,584 B | 单 expert gate+up 权重大小（INT4，2×2048×1408×0.5）|
| `WBYTES_S3` | 1,441,792 B | 单 expert down 权重大小（INT4，1×2048×1408×0.5）|
| `MAX_BW` | 128 B/cc | HBM → L1 最大聚合带宽 |
| `EXACT_TAIL_MAX_TOKENS` | 4 | 尾部 ≤2 个 expert 且总 token ≤4 时才用精确枚举估价 |

### 1.2 Shape（矩阵分块规格）

每个 expert 计算分两段 DMA pipeline：

```
│◄── S1(gate/up) ──►│◄── S2(compute) ──►│◄── S3(down) ──►│◄── S4(compute) ──►│
  DMA 搬入权重        CPU/加速器计算        DMA 搬入权重       CPU/加速器计算
```

`Shape` 决定"每次 DMA 搬多少行"（M_dim）和"需要多少带宽"（bw_req/alloc）：

```c
static const shape_def_t SHAPE[3] = {
    /* A */ {m=8, bw=32,  alloc=64,  T_s1=90112, T_s3=45056, t_dma_s1=45056, t_dma_s3=22528},
    /* B */ {m=4, bw=64,  alloc=64,  T_s1=45056, T_s3=22528, t_dma_s1=45056, t_dma_s3=22528},
    /* C */ {m=2, bw=128, alloc=128, T_s1=22528, T_s3=11264, t_dma_s1=22528, t_dma_s3=11264},
};
```

- **ShapeA**（M=8）：DMA 一次搬 8 行，需 32 B/cc，DMA 比计算慢——适合 token 多、单侧独占时。
- **ShapeB**（M=4）：DMA 一次搬 4 行，需 64 B/cc，DMA 与计算均衡——并发首选。
- **ShapeC**（M=2）：DMA 一次搬 2 行，需 128 B/cc——DMA 最快，独占全带宽时最优。

**关键约束**：两个 cluster 同时做 S1/S3 DMA 时，`alloc_C2 + alloc_C3 ≤ 128`，所以并发时只能选 ShapeA 或 ShapeB（alloc=64）。

---

## 2. 辅助计算函数

### `udiv_ceil(a, b)` — 整数向上除法

```c
static inline uint32_t udiv_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}
```

用于计算"需要几次 DMA 迭代"。

---

### `best_s2_compute(remaining)` / `best_s4_compute(remaining)` — 最优计算时间

```c
static inline uint32_t best_s2_compute(uint32_t remaining) {
    if (remaining == 0) return 0;
    return udiv_ceil(remaining, SHAPE_C_M_DIM) * SHAPE[MOE_SHAPE_C].T_s1;
}
```

S2（gate/up 计算）剩余 `remaining` 个 token 时，最快跑完需要多少 cc：  
用 ShapeC（M_dim=2）切分，每段 `T_s1 = 22528 cc`，共 `⌈remaining/2⌉` 段。  
对应 Python `_best_s2_compute`。

---

### `task_time_for_shapes(ntok, s1, s3)` — 给定 shape 组合的总任务时间

```c
static uint32_t task_time_for_shapes(uint32_t ntok, int s1, int s3) {
    uint32_t rem_s2 = (ntok > SHAPE[s1].m_dim) ? (ntok - SHAPE[s1].m_dim) : 0u;
    uint32_t rem_s4 = (ntok > SHAPE[s3].m_dim) ? (ntok - SHAPE[s3].m_dim) : 0u;
    return SHAPE[s1].T_s1 + best_s2_compute(rem_s2)
         + SHAPE[s3].T_s3 + best_s4_compute(rem_s4);
}
```

计算公式：
$$T_{task} = T_{s1} + T_{s2,rest} + T_{s3} + T_{s4,rest}$$

第一段 DMA+计算（S1）跑 `m_dim` 个 token，剩余 `ntok - m_dim` 个在 S2 继续计算。

---

### `best_task_time(ntok)` — 无带宽约束时的任务最短时间

```c
static uint32_t best_task_time(uint32_t ntok) {
    // 穷举 3×3=9 种 (s1,s3) 组合，取最小
    for s1 in {A,B,C}:
        for s3 in {A,B,C}:
            best = min(best, task_time_for_shapes(ntok, s1, s3))
}
```

对应 Python `_best_task_time`。用于 **单侧独占带宽**（如最后一个 expert、SOLO 执行）时的下界估计。

---

### `best_concurrent_task_time(ntok)` — 并发约束下的任务最短时间

```c
static uint32_t best_concurrent_task_time(uint32_t ntok) {
    // 只考虑 alloc ≤ 64 的 shape（并发时双侧各占 64 B/cc）
    for s1 in {A,B}:   // ShapeC alloc=128 > 64，不合法
        for s3 in {A,B}:
            best = min(...)
}
```

对应 Python `_best_concurrent_task_time`。用于 **两个 cluster 同时运行** 时的下界估计。

---

### `best_solo_shape_s1/s3(ntok)` — 独占时最优 S1/S3 shape

```c
static moe_shape_t best_solo_shape_s1(uint32_t ntok) {
    // 最小化 T_s1 + best_s2_compute(ntok - m_dim)
}
```

在"不考虑带宽竞争"的情况下，返回使 S1+S2 时间最短的 shape。  
token 少时选 ShapeC（DMA 快），token 多时可能选 ShapeA（M_dim 大，减少分段次数）。

---

### `best_conc_shape_s1/s3(ntok)` — 并发时最优 shape

```c
static moe_shape_t best_conc_shape_s1(uint32_t ntok) {
    // alloc ≤ MAX_BW/2 = 64，即只考虑 ShapeA 和 ShapeB
}
```

并发约束下，两侧各最多占 64 B/cc，ShapeC（alloc=128）不合法。

---

## 3. 核心数据结构：`snap_t`（Cluster 状态快照）

`snap_t` 完整记录一个 cluster 在某个 task 执行后的所有时间戳，对应 Python `FourStageSnap`：

```c
typedef struct {
    uint32_t task_start;   // 任务开始时刻
    uint32_t task_end;     // 任务结束时刻（dma3_end + S4计算）
    uint32_t dma1_end;     // S1 DMA 结束时刻
    uint32_t s1_end;       // S1 全段（DMA+首批计算）结束时刻
    uint32_t s2_end;       // S2 计算（所有剩余token）结束时刻
    uint32_t dma3_end;     // S3 DMA 结束时刻
    uint32_t s3_end;       // S3 全段结束时刻

    // 预取（next-S1 prefetch）状态
    int32_t  pf_start;     // prefetch DMA 开始时刻，-1=无
    int32_t  pf_end;       // prefetch DMA 结束时刻，-1=无，0=初始已缓存
    int16_t  pf_eid;       // 被预取的 expert ID，-1=无
    uint16_t pf_bw;        // prefetch DMA 带宽占用
    uint8_t  pf_full;      // 1=S1+S3 全部缓存（初始 resident），0=仅 S1

    // DMA 带宽
    uint16_t bw_s1;        // S1 DMA 期间占用的带宽
    uint16_t bw_s3;        // S3 DMA 期间占用的带宽

    int16_t  cur_eid;      // 当前 task 的 expert ID，-1=空闲
    uint8_t  shape_s1;     // S1 shape 枚举值
    uint8_t  shape_s3;     // S3 shape 枚举值
    uint16_t ntok;         // token 数
    uint8_t  skip_dma_s1;  // S1 权重已缓存，跳过 DMA
    uint8_t  skip_dma_s3;  // S3 权重已缓存，跳过 DMA

    // S2 down-prefetch（将 S3 DMA 提前到 S2 计算期间）
    int32_t  s2pf_start;   // S2 prefetch 开始，-1=无
    int32_t  s2pf_end;     // S2 prefetch 结束
    uint16_t s2pf_bw;      // S2 prefetch 带宽

    uint8_t  is_wait;      // 1=合成等待快照（无实际 task）
} snap_t;
```

### 时间轴示例（无缓存，ShapeB，ntok=6）

```
t_start                      dma1_end=s1_end  dma3_end=s3_end   task_end
   │                               │                │                │
   ├── S1 DMA (45056cc,64B/cc) ────┤                │                │
   ├── S1 compute first 4tok──────►├── S2 remain ──►│                │
                                   │── S3 DMA ──────►│               │
                                                     │── S4 remain ──►│
```

---

## 4. Snap 构造函数

### `snap_make_initial(s, cached_eid)` — 初始快照

```c
static void snap_make_initial(snap_t *s, int16_t cached_eid) {
    // 所有时间戳置零，cur_eid = -1（空闲）
    // 若 cached_eid >= 0：pf_eid = cached_eid，pf_end = 0，pf_full = 1
    // 含义：该 expert 的 S1+S3 权重在 t=0 已经预存在 L1 中
}
```

对应 Python `make_initial_snap(cached_eid)`。  
**关键设计**：初始缓存不是通过 `cur_eid` 表达，而是通过 `pf_eid/pf_end/pf_full` 字段。这样 `cache_hit()` 函数可以统一处理"预取命中"和"初始驻留"两种情况。

---

### `snap_assign(out, start, s1, s3, ntok, eid, cache_flags)` — 任务分配快照

这是最核心的构造函数，对应 Python `FourStageSnap.from_assign()`：

```c
static void snap_assign(snap_t *out, uint32_t start,
                        moe_shape_t s1, moe_shape_t s3,
                        uint16_t ntok, int16_t eid, uint8_t cache_flags) {
    // cache_flags: CACHE_S1_READY(bit0) | CACHE_S3_READY(bit1)
    // 若 S1 已缓存：dma1_end = start（无 DMA），bw_s1 = 0
    // 若 S3 已缓存：dma3_end = s2_end（无 DMA），bw_s3 = 0

    uint32_t gate_end = start + T_s1 + best_s2_compute(ntok - m_dim_s1);
    uint32_t down_end = gate_end + T_s3 + best_s4_compute(ntok - m_dim_s3);

    // S1 缓存命中时：compute 照常跑，只是 DMA 部分跳过
    out->dma1_end = s1_cached ? start : (start + t_dma_s1);
    out->bw_s1    = s1_cached ? 0 : S1->alloc;
    // 同理 S3
}
```

**注意**：`from_assign` 会**清空**所有预取字段（pf_eid=-1 等），因为新 task 开始时预取状态重置。

---

### `snap_wait(out, t)` — 合成等待快照

```c
static void snap_wait(snap_t *out, uint32_t t) {
    // 所有时间戳 = t，cur_eid = -1，bw = 0，is_wait = 1
}
```

用于 WAIT-PAIR 策略：当一个 cluster 先发了一个小 expert，另一个 cluster 需要"等到时刻 t 才能开始"——用 `snap_wait(t)` 表示该等待状态。

---

## 5. 带宽计算

### `snap_active_bw(s, t)` — t 时刻该 cluster 占用的带宽

```c
static inline uint32_t snap_active_bw(const snap_t *s, uint32_t t) {
    uint32_t bw = 0;
    // 主任务 S1 DMA 期间：[task_start, dma1_end)
    if (cur_eid >= 0 && t < task_end) {
        if (t in [task_start, dma1_end)  )  bw = bw_s1;
        if (t in [s2_end,     dma3_end)  )  bw += bw_s3;
    }
    // Next-S1 prefetch DMA 期间
    if (pf_start >= 0 && t in [pf_start, pf_end)) bw += pf_bw;
    // S2 down-prefetch DMA 期间
    if (s2pf_start >= 0 && t in [s2pf_start, s2pf_end)) bw += s2pf_bw;
    return bw;
}
```

对应 Python `active_bw_at(t)`。三种 DMA 可能同时占用带宽：主任务 S1/S3 + 预取 + S2 down-prefetch。

---

### `bw_feasible(a, b)` — 两 cluster 带宽可行性检验

```c
static int bw_feasible(const snap_t *a, const snap_t *b) {
    // 收集所有时间变化点（最多 22 个）
    // 在每个区间 [pts[i], pts[i+1]) 采样一次
    // 若 active_bw(a,t) + active_bw(b,t) > 128 则不可行
}
```

这是 **BW 约束核心检验**，对应 Python `bw_feasible()`。  
原理：带宽是分段常数函数，只需在每个变化点处检查一次即可。  
变化点来源：S1 DMA 开始/结束、S3 DMA 开始/结束、预取开始/结束，最多 11×2=22 个点。

---

## 6. S2 Down-Prefetch（将 S3 DMA 提前）

### 背景

正常流程中 S3 DMA 在 S2 计算结束后才开始。但 S2 计算期间如果 S1 DMA 已经结束（带宽空闲），可以**提前**搬入 S3 权重——这叫 **S2 down-prefetch**：

```
正常:  │── S1 DMA ──│── S2 ──│── S3 DMA ──│── S4 ──│
提前:  │── S1 DMA ──│── S2 ──│            │── S4 ──│
                    │ S3 DMA（在 S2 期间提前做完）│
```

这样 S3 DMA 和 S4 计算可以更早开始，缩短 task_end。

---

### `collect_s2_down_prefetch_starts(sn, s3, peer, out)` — 候选提前开始时刻

```c
static int collect_s2_down_prefetch_starts(...) {
    // 有效窗口：[task_start, s2_end - t_dma_s3]
    // 候选时刻：窗口端点 + 自身 bw_change_pts + peer 的 bw_change_pts
    //           + 上述各点对齐后退 t_dma_s3 的点
    // 去重并排序后返回
}
```

目的：找出所有"可能最优"的 S2 prefetch 开始时刻，避免穷举所有时刻。

---

### `snap_apply_s2_down_prefetch(sn, s3, start)` — 应用 S2 prefetch

```c
static void snap_apply_s2_down_prefetch(snap_t *sn, moe_shape_t s3, uint32_t start) {
    sn->s2pf_start = start;
    sn->s2pf_end   = start + SHAPE[s3].t_dma_s3;
    sn->s2pf_bw    = SHAPE[s3].alloc;
    sn->dma3_end   = sn->s2_end;   // S3 DMA 已提前做完，task 进入 S3 段时 DMA 已就绪
    sn->bw_s3      = 0;            // 主 S3 DMA 不再占带宽
    sn->skip_dma_s3 = 1;
}
```

---

### `apply_s2_down_prefetch_pair(a, s3a, b, s3b)` — 双侧联合优化

```c
static void apply_s2_down_prefetch_pair(...) {
    // 枚举 (start_a, start_b) 的所有候选组合（含"不提前"选项）
    // 选 BW 可行 + 提前个数最多（score=2>1>0）+ 启动时刻最早（start_sum 最小）的组合
}
```

两个 cluster 同时有 S2 prefetch 机会时，需要**联合**优化，因为各自的提前 DMA 会互相影响带宽预算。

---

## 7. Next-S1 Prefetch（S4 期间预取下个 expert 的 S1 权重）

### 背景

当前 expert 运行到 S4 阶段时，S3 DMA 已结束，带宽空闲。可以利用这段时间**提前搬入下一个 expert 的 gate+up 权重**（S1 权重），使下个任务开始时 S1 DMA 已完成：

```
当前 task:  │── S1 DMA ──│── S2 ──│── S3 DMA ──│── S4 ──│
                                                  ↑ 在这里偷空搬入 next expert S1
下个 task:  [S1 已就绪，直接 S2 compute]
```

---

### `collect_next_s1_prefetch_starts(sn, pf_shape, peer, out)` — 候选开始时刻

```c
// 有效窗口：[s2_end, task_end]（S4 期间）
// 候选：s2_end / dma3_end / s3_end / task_end 及 bw_change_pts 对齐点
```

---

### `snap_apply_next_s1_prefetch(sn, next_eid, pf_shape, start)` — 应用 S4 prefetch

```c
sn->pf_start = start;
sn->pf_end   = start + SHAPE[pf_shape].t_dma_s1;
sn->pf_eid   = next_eid;
sn->pf_bw    = SHAPE[pf_shape].alloc;
sn->pf_full  = 0;   // 只预取 S1，不是 full cache
```

---

### `collect_next_s1_prefetch_snaps(sn, peer, next_eid, out, n)` — 枚举候选 snap

对三种 shape 各收集候选启动时刻，每种 shape 取第一个 BW 可行的时刻，最多生成 8 个候选 snap（含原始"不预取"版本）。

---

### `apply_next_s1_prefetch_pair(a, b, next_eid)` — 双侧联合优化

```c
static void apply_next_s1_prefetch_pair(snap_t *a, snap_t *b, int16_t next_eid) {
    // 为 a 枚举最多 8 个候选（含无预取），为 b 枚举最多 8 个候选
    // 选 BW 可行 + 预取成功侧数最多 + pf_end 最小的组合
}
```

**在主循环每次迭代最开始调用**：在评估当前 top0 之前，先尝试在上一轮 snap 上做 next-S1 预取。这是调度器的**关键前处理**，等同于 Python `with_optional_next_s1_prefetch_pair()`。

---

## 8. 缓存命中检测

### `cache_hit(cl, eid, t)` — 在时刻 t 检查 expert eid 是否已缓存

```c
static inline uint8_t cache_hit(const snap_t *cl, int16_t eid, uint32_t t) {
    if (cl->pf_eid != eid) return 0;   // 不是目标 expert
    if (cl->pf_end < 0) return 0;      // 没有预取记录
    if ((uint32_t)cl->pf_end > t) return 0;  // 预取还没完成
    // pf_full=1 表示 S1+S3 全部缓存（初始驻留），=0 只有 S1
    return CACHE_S1_READY | (pf_full ? CACHE_S3_READY : 0);
}
```

**设计要点**：无论是"初始缓存"（pf_end=0）还是"S4 prefetch"（pf_end>0），都通过 pf_* 字段统一表达，`cache_hit()` 只需检查 `pf_end ≤ t`。

返回值是 bit mask：`CACHE_S1_READY | CACHE_S3_READY`，传给 `snap_assign()` 的 `cache_flags` 参数，决定跳过哪些 DMA。

---

## 9. 代价估算函数

### `greedy_heuristic(c2_end, c3_end, rem_ntok, n_rem)` — 多剩余 expert 下界估计

```c
static uint32_t greedy_heuristic(...) {
    if (n_rem == 0) return max(c2_end, c3_end);
    
    if (n_rem == 1) {
        // 精确下界：考虑 SOLO 和 SPLIT 两种方案
        t_early = min(c2_end, c3_end);
        t_late  = max(c2_end, c3_end);
        solo_cost  = max(t_late, t_early + best_task_time(ntok));
        split_cost = max(t_late, t_early + best_concurrent_task_time(ntok/2));
        return min(solo_cost, split_cost);
    }
    
    // n_rem >= 2：乐观估计
    // base = max(c2_end, c3_end)（同步点）
    // extra = max(最长单任务时间, 所有任务之和/2)（至少需要这么多额外时间）
    base  = max(c2_end, c3_end);
    extra = max(max_task_time, sum_task_time / 2);
    return base + extra;
}
```

对应 Python `_greedy_heuristic`。这是一个**乐观下界**（actual ≥ heuristic），用于剪枝不优的候选。

---

### `sim1(c2, c3, eid, ntok)` — 最后一个 expert 的精确模拟

这是最耗计算的函数，对最后一个 expert 做**穷举最优**：

```c
static uint32_t sim1(const snap_t *c2, const snap_t *c3, int16_t eid, uint16_t ntok) {
    // 步骤0：先尝试 next-S1 prefetch（在当前 snap 上提前预取）
    apply_next_s1_prefetch_pair(&c2_local, &c3_local, eid);
    
    // 方法 A：等两侧都空闲（t = max(t2,t3)），SOLO 或 SPLIT
    for s1, s3 in all 3×3 shapes:
        SOLO on C2 (use c2 cache)  → cost = sn.task_end
        SOLO on C3 (use c3 cache)  → cost = sn.task_end
    if ntok >= 2:
        SPLIT: 枚举切割点 {ceil/floor halves}，两侧各自枚举 3×3 shapes
               apply_s2_down_prefetch_pair + bw_feasible 检验

    // 方法 B：空闲侧提前启动（不等另一侧）
    if t2 != t3:
        idle_cl = 较早完成的那侧
        try_starts = idle_t 以及 busy 侧所有 bw_change_pts
        for each t_start:
            for s1, s3:
                在 t_start 启动，apply_s2_down_prefetch，bw_feasible 检验
}
```

**方法 B 的物理意义**：busy 侧 S1 DMA 结束后带宽释放，idle 侧可以立刻用高带宽（甚至 ShapeC）启动，比等 `now` 更早完成。

---

### `eval_cost(sna, snb, rem_ntok, n_rem, last_eid)` — 综合代价函数

```c
static uint32_t eval_cost(...) {
    if (n_rem == 0) return max(sna->task_end, snb->task_end);
    if (n_rem == 1) return sim1(sna, snb, last_eid, rem_ntok[0]);  // 精确
    if (n_rem == 2 && total_ntok <= EXACT_TAIL_MAX_TOKENS) {
        // 精确双 expert 估计（两个都很小时）
        t_early = min(sna->end, snb->end);
        t_late  = max(sna->end, snb->end);
        solo_seq = t_early + sum(best_task_time each);
        pair_after_late = t_late + max(best_concurrent_task_time each);
        return min(max(t_late, solo_seq), pair_after_late);
    }
    return greedy_heuristic(sna->end, snb->end, rem_ntok, n_rem);
}
```

**分层精度**：
- 0 remaining → 直接读 max(task_end)
- 1 remaining → `sim1` 精确穷举
- 2 remaining 且 token ≤ 4 → 简化精确估计
- 其他 → `greedy_heuristic` 乐观下界

---

### `split_hot_tail_cost(sna, snb, new_rem, n_rem)` — 热 expert SPLIT 的精确代价

当剩余列表形如 `[hot_expert(多 token), tail1(1 tok), tail2(1 tok)]` 时，  
贪心 heuristic 会低估：它不知道 hot expert 可以 SPLIT 给两侧同时跑。

此函数专门处理这种情况：

```c
static uint32_t split_hot_tail_cost(...) {
    // 条件：hot_ntok >= 2，尾部 ≤ 2 个 expert
    // 步骤1：先做 next-S1 prefetch（针对 hot_eid）
    apply_next_s1_prefetch_pair(&c2_hot, &c3_hot, hot_eid);
    // 只有两侧都在同一时刻空闲时才能做 SPLIT
    if (c2_hot.task_end != c3_hot.task_end) return eval_cost_entries(...);

    // 步骤2：枚举所有切割点（ceil/floor + shape.M_dim 对齐）
    for cut_A in cuts:
        cut_B = hot_ntok - cut_A
        for s1_c2, s1_c3, s3_c2, s3_c3:  // 3^4 = 81 种组合
            snap_assign C2: cut_A tokens of hot_eid
            snap_assign C3: cut_B tokens of hot_eid
            apply_s2_down_prefetch_pair
            if bw_feasible:
                cost = eval_cost_entries(tail_rem)  // 评估尾部
                best = min(best, cost)
}
```

**使用场景**：在主循环的 WAIT-SINGLE-PAIR 候选计算 `cost` 时调用。

---

## 10. 候选追踪器：`cand_t`

```c
typedef struct {
    uint32_t cost;          // 总 makespan 代价
    uint32_t snap_min;      // min(c2_after.end, c3_after.end)，平局时的次级排序
    uint32_t snap_max;      // max(c2_after.end, c3_after.end)，平局时的三级排序
    snap_t   c2_after;      // 本步决策后 C2 的状态
    snap_t   c3_after;      // 本步决策后 C3 的状态
    snap_t   pre_emit;      // WAIT-SINGLE-PAIR 时需要先发出的那个 task 的 snap
    int16_t  consumed_eid_a/b/c;  // 本步消耗的 expert IDs（最多 3 个）
    moe_cluster_t pre_cluster;    // pre_emit 发往哪个 cluster
    uint8_t  emit_pre;     // 是否需要发出 pre_emit
    uint8_t  emit_c2;      // c2_after 是否有新 task
    uint8_t  emit_c3;      // c3_after 是否有新 task
} cand_t;
```

---

### `cand_better(best, cost, c2_after, c3_after)` — 比较两个候选

```c
static int cand_better(const cand_t *best, uint32_t cost, ...) {
    if (cost != best->cost) return cost < best->cost;          // 首选：代价更低
    if (snap_max != best->snap_max) return snap_max < best->snap_max; // 平局：较早完成的更晚侧
    return snap_min < best->snap_min;                          // 再平局：较早完成的更早侧
}
```

三级排序确保确定性（tie-break）。对应 Python `cand_better`。

---

### `cand_update(...)` — 更新最优候选

记录当前最优的 (c2_after, c3_after, consumed_eids, emit_flags)。

---

### `cand_set_pre_emit(best, cluster, pre_emit)` — 记录前置 task

用于 WAIT-SINGLE-PAIR：该策略先发一个小 expert（`pre_emit`），然后两侧 both_idle，再继续后续决策。`pre_emit` 需要单独记录以便在应用决策时发出。

---

## 11. 主调度循环：`moe_schedule_impl(req, out, initial_cache_mask)`

这是算法的核心，对应 Python `analytical_schedule()` 主循环。

### 11.1 初始化

```c
// 1. 将 req->experts[] 拷贝到 rem[]，按 ntok 降序排列（最多 token 的排最前）
sort_desc(rem, nrem);

// 2. 初始化 C2/C3 快照（考虑 initial_cache_mask 决定哪些 expert 已缓存）
snap_make_initial(&c2, req->cache_eid_c2);
snap_make_initial(&c3, req->cache_eid_c3);
```

### 11.2 初始缓存命中处理

```c
if (c2_cached_idx >= 0 && c3_cached_idx >= 0 && cache_eid_c2 == cache_eid_c3) {
    // 两侧缓存同一个 expert：SPLIT 后立即发出（两侧各跑一半）
} else {
    if (c2_cached_idx >= 0) { 立刻发出该 expert 到 C2; remove from rem; }
    if (c3_cached_idx >= 0) { 立刻发出该 expert 到 C3; remove from rem; }
}
```

对应 Python `_allow_initial_cache_choices` 分支：缓存的 expert 先行发出，不参与后续贪心决策。

### 11.3 主循环

每次迭代处理 rem[0]（token 数最多的剩余 expert，称为 top0）：

```
while (nrem > 0):
    apply_next_s1_prefetch_pair(c2, c3, rem[0].eid)  // 尝试在上一轮 snap 上预取 top0
    t2 = c2.task_end; t3 = c3.task_end
    now = max(t2, t3); both_idle = (t2 == t3)
    
    if n == 1:      → 精确最优（穷举 Method-A/B）
    if !both_idle:  → 简单 SOLO（在空闲侧发 top0）
    if both_idle:   → 全搜索（PAIR + SPLIT + WAIT）
```

---

### 11.4 分支 A：`n == 1`（只剩最后一个 expert）

**等价于调用 `sim1`，但直接内联以能 emit task**：

```
方法 A：等两侧都空闲（t=now）：
    枚举 SOLO C2 / SOLO C3 / SPLIT（完整切割点集合）
    全 3×3 shape 组合，apply_s2_down_prefetch_pair
    取 max(task_end, 对侧 task_end) 最小的

方法 B：空闲侧提前启动：
    try_starts = {idle_t} ∪ busy.bw_change_pts()
    枚举每个 t_start，3×3 shape，bw_feasible 检验
    取 max(sn.task_end, busy.task_end) 最小的

选 best，emit task，nrem--
```

---

### 11.5 分支 B：`n >= 2, !both_idle`（一侧空闲，一侧忙）

**策略：在空闲侧发 top0（最大 expert），尽量早启动**：

```
idle_t = 空闲侧 task_end
try_starts = {idle_t} ∪ busy.bw_change_pts()

for t_start in try_starts:
    for s1, s3:  // 3×3 shapes
        snap_assign(idle_side, t_start, ...)
        apply_s2_down_prefetch_single(sn, s3, busy_cl)
        if bw_feasible: cost = max(sn.end, busy.end) → 取最小

emit task, nrem--
```

**为何遍历 bw_change_pts**：busy 侧 DMA 结束后带宽释放，空闲侧可以切换到更高带宽的 shape（如 ShapeC），所以更晚启动可能反而更快完成。

---

### 11.6 分支 C：`n >= 2, both_idle`（两侧都空闲，主决策点）

这是最复杂的部分。用 `cand_t best` 追踪最优候选，评估以下候选族：

---

#### 候选族 1：PAIR(top0, topK)，K=1..min(n-1,3)

```
for K in {1,2,3} (候选配对 expert):
    for dir in {0,1}:  // top0→C2 + topK→C3，或反过来
        for s1a, s1b, s3a, s3b:  // 3^4 = 81 种 shape 组合
            // BW 剪枝：两侧都无缓存且 alloc > 64，直接跳过
            if !cache and alloc > MAX_BW/2: continue

            snap_assign C2, snap_assign C3
            apply_s2_down_prefetch_pair
            if bw_feasible:
                n_rem = nrem - 2（去掉 top0 和 topK）
                cost  = eval_cost(remaining)
                cand_update if better
```

**物理意义**：PAIR 是最常见的决策——两个 expert 同时发到两个 cluster 并行跑。K=1 是最典型的（token 最多的两个配对），K=2,3 允许跳过大 expert 先配对较小的。

---

#### 候选族 2：PAIR(topK, topJ)，K<J，n≥3（跳过 top0）

```
for K in {1,2}, J in {K+1..3}:
    // 延迟 top0，先处理两个较小的 expert
    // S3 只枚举 {ShapeB, ShapeC}（加速，Python 同）
    if cand_better: cand_update（消耗 eK, eJ）
```

**物理意义**：top0 很大（需要 SPLIT），先让两个小 expert 占满两侧，等它们做完再对 top0 做精确 SPLIT——`eval_cost` 用 `sim1` 计算这个 SPLIT 的精确代价。

---

#### 候选族 3：SPLIT(top0)

```
cuts = {ceil(n0/2), floor(n0/2), M_dim for each shape, n0 - M_dim for each shape}
for cut_A in cuts:
    cut_B = n0 - cut_A
    for s1a, s1b, s3a, s3b:  // 81 种
        snap_assign C2: cut_A tokens of top0
        snap_assign C3: cut_B tokens of top0
        apply_s2_down_prefetch_pair
        if bw_feasible:
            cost = eval_cost(remaining without top0)
            cand_update if better
```

**物理意义**：top0 token 数很多（远超单侧最优），切分成两半并行跑，减少总 makespan。切割点不只有一半——有时切 `M_dim`（对齐 DMA 迭代边界）更优。

---

#### 候选族 4：WAIT-SINGLE-PAIR（n≥5）

```
for K in {1..4}:  // 选一个 1-token 的小 expert 先单独发
    if rem[K].ntok != 1: continue
    first_sn = snap_assign(now, s1_first, s3_first, eid_K)  // 单发到 C2
    t_pair = first_sn.task_end
    wait_sn = snap_wait(t_pair)  // C3 等待到 t_pair
    
    for pair candidates (anchor, cand) where cand.ntok==1:
        // 在 t_pair 时刻两侧 both_idle，做一对 PAIR
        snap_assign C2: anchor at t_pair
        snap_assign C3: cand at t_pair
        apply_s2_down_prefetch_pair
        if bw_feasible:
            cost = split_hot_tail_cost(rem_after_pair)  // 评估尾部（考虑 hot SPLIT）
            cand_update + cand_set_pre_emit(first_sn)
```

**物理意义**：在 token 分布极不均匀时（如 [17,1,1,1,1]），先用一个 SINGLE 消耗一个冷 expert，然后两侧都空闲再 PAIR 两个冷 expert，这样为后续的热 expert SPLIT 创造"两侧同步"的最优条件。

---

#### 候选族 5：WAIT-PAIR（K=1..3，拖延 top0）

```
for K in {1,2,3}:
    for dir in {C2, C3}:  // topK 发到哪侧
        for s1, s3:  // 3×3 shapes
            sn_k = SOLO(topK) on dir at now
            wait  = snap_wait(sn_k.task_end)  // 另一侧等待
            if bw_feasible:
                n_rem = nrem - 1（去掉 eK）
                cost  = eval_cost(remaining without topK)
                // eval_cost 会在下一步看到 both_idle 状态，
                // 并对 top0 做 SPLIT 或 PAIR
                cand_update if better
```

**物理意义**：top0 很大，需要 SPLIT，但当前两侧并不同时空闲——先发一个小 expert topK，等它结束后两侧都空闲，然后对 top0 做 SPLIT。`eval_cost` 的 `sim1` 路径会精确计算这个 SPLIT 的代价。

---

### 11.7 应用最优候选

```c
if (best.emit_pre) emit_task(pre_cluster, &best.pre_emit);  // WAIT-SINGLE-PAIR 的先发 task
c2 = best.c2_after;
c3 = best.c3_after;
if (best.emit_c2) emit_task(C2, &c2);
if (best.emit_c3) emit_task(C3, &c3);
// 从 rem[] 中删除已消耗的 expert
```

---

### 11.8 后处理：填充 prefetch_eid

```c
// 对每个 task，找同一 cluster 上的下一个 task（expert_id 不同）
// 记录为 prefetch_eid，用于提示 DMA 控制器做 S4 next-S1 prefetch
for i in tasks:
    for j in (i+1)..tasks:
        if tasks[j].cluster == tasks[i].cluster and tasks[j].eid != tasks[i].eid:
            tasks[i].prefetch_eid = tasks[j].eid
            break
```

注意：这只是**建议性提示**，当前调度的时序没有依赖它（没有假设 prefetch 一定会命中）。

---

## 12. 外层接口：`moe_schedule(req, out)` — 缓存掩码剪枝

```c
moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out) {
    // 检查 cache_eid_c2/c3 是否实际在请求的 expert 列表中
    if (cache_eid_c2 在 req.experts 中) valid |= INIT_CACHE_C2;
    if (cache_eid_c3 在 req.experts 中) valid |= INIT_CACHE_C3;

    // 构建候选掩码列表（至多 4 个）：
    masks = [0]                              // 不启用任何初始缓存
    if C2 有效: masks += [INIT_CACHE_C2]    // 只启用 C2 缓存
    if C3 有效: masks += [INIT_CACHE_C3]    // 只启用 C3 缓存
    if 两者都有效: masks += [C2|C3]         // 两者都启用

    // 对每个掩码运行一次 moe_schedule_impl，取 makespan 最小的
    for mask in masks:
        moe_schedule_impl(req, &candidate, mask)
        if candidate.makespan < best.makespan: best = candidate
}
```

**为何需要多掩码**：初始缓存是"可选优化"——某些情况下（如 token 分布与缓存 expert 不匹配），忽略缓存、用正常调度可能更优。用 1-4 次调用（而不是 1 次）来确保找到全局最优。

**加速效果**：从 Python 完整枚举（会对缓存 expert 做多路初始 snap）到 C 的掩码剪枝，运行时从 ~78.5ms 降到 ~20.9ms/call（server x86 -O2）。

---

## 13. 辅助工具函数

### `sort_desc(e, n)` — 按 ntok 降序排列

插入排序，expert 按 token 数从多到少排列。确保主循环每步处理 token 最多的 expert（贪心核心假设）。

### `rem_excluding(src, n, skip_a, skip_b, out_ntok, out_last_eid)` — 过滤剩余列表

从 rem[] 中去掉指定的 expert，返回剩余的 ntok 数组（用于 `eval_cost`）和最后一个 eid（用于 `sim1`）。

### `rem_excluding_entries(...)` — 过滤并保留完整 entry

类似 `rem_excluding` 但保留 eid 字段，用于 WAIT-SINGLE-PAIR 等需要完整信息的场景。

### `snap_assign_best_cached(out, start, eid, ntok)` — 最优全缓存 snap

枚举所有 shape 组合，假设 S1+S3 全部缓存（`cache_flags = S1|S3`），返回 task_end 最小的 snap。用于"初始缓存 expert 立即发出"的路径。

### `eval_cost_entries(sna, snb, rem, n_rem)` — eval_cost 的 entry 版本

将 `entry_t[]` 转换为 `uint16_t ntok[]` 后调用 `eval_cost`。

### `add_cut(cuts, n, cut, ntok)` — 向切割点集合添加（去重）

`split_hot_tail_cost` 使用，最多 8 个切割点。

---

## 14. 调度算法决策树总结

```
每步迭代（rem 非空）
│
├─ 前处理：apply_next_s1_prefetch_pair(top0)
│
├─ [n==1] 精确穷举最后一个 expert
│     ├─ 方法A：等两侧都空闲，SOLO/SPLIT，枚举所有 shape
│     └─ 方法B：空闲侧提前启动，枚举 bw_change_pts × shape
│
├─ [n≥2, !both_idle] 简单 SOLO
│     └─ 空闲侧启动 top0，枚举 bw_change_pts × shape
│
└─ [n≥2, both_idle] 全搜索（取 cost 最小）
      ├─ PAIR(top0, topK)，K=1..3，两方向，81种 shape
      ├─ PAIR(topK, topJ)，K<J，跳过 top0，81种 shape（S3限{B,C}）
      ├─ SPLIT(top0)，完整切割点集，81种 shape
      ├─ WAIT-SINGLE-PAIR（n≥5），先发 1-token 冷 expert，再双侧 PAIR
      └─ WAIT-PAIR，先 SOLO topK，等两侧 both_idle，eval_cost 用 sim1 精确估计
```

---

## 15. 关键设计权衡（PPA 视角）

| 设计点 | 选择 | 原因 |
|---|---|---|
| 不做 beam-search | 贪心 + 精确 1-step lookahead | 减少计算量 O(N) vs O(N²)|
| `sim1` 精确枚举 | 只在最后一个 expert 时调用 | 计算量 O(81×cuts) 可接受 |
| `split_hot_tail_cost` | 只有少量尾部时调用 | 精确 vs 速度的平衡 |
| `EXACT_TAIL_MAX_TOKENS=4` | 两个 expert 总 token ≤4 时精确 | 小 token 场景误差大，精确更重要 |
| S3 只枚举 {B,C} | PAIR(topK,topJ) 场景 | S3=A 极少最优，减少 50% 计算 |
| 掩码剪枝（最多 4 次调用）| moe_schedule 外层 | 避免冗余 run（无缓存时只跑 1 次）|
