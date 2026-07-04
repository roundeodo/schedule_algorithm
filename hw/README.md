# MoE Hardware Scheduler – Design Notes

## Overview

`moe_hw_scheduler.sv` implements a synthesisable hardware scheduler that
dispatches Mixture-of-Experts (MoE) experts to two compute clusters (C2, C3)
with near-optimal makespan, matching the behaviour of the Python beam-search
scheduler (`four_stage_scheduler.py`).

---

## File List

| File | Description |
|------|-------------|
| `moe_hw_scheduler.sv`  | Main scheduler module (FSM + combinational DT) |
| `tb_moe_hw_scheduler.sv` | Self-checking testbench (6 directed tests) |

---

## Architecture

```
 Top-K Router ──►┌─────────────────────────────────────────────────────┐
                 │              moe_hw_scheduler                        │
 C2/C3 cache ──►│                                                       │──► C2 assignment
                 │  ┌──────────────────────┐  ┌────────────────────┐   │    (eid, ntok,
 C2/C3 done  ──►│  │ Action selector DT   │  │  Shape LUT         │   │    shape_s1,
                 │  │ (combinational,      │  │  (combinational,   │   │    shape_s3,
                 │  │  depth 4)            │  │   analytical)      │   │    cached)
                 │  └──────────────────────┘  └────────────────────┘   │
                 │        │                         │                   │──► C3 assignment
                 │  ┌─────▼─────────────────────────▼─────────────┐    │
                 │  │        Schedule FSM  (IDLE/ASSIGN/WAIT/DONE) │    │
                 │  │   Expert queue (8-deep shift register)       │    │
                 │  └─────────────────────────────────────────────-┘    │
                 └─────────────────────────────────────────────────────┘
```

---

## Interface

### Inputs
| Signal | Width | Description |
|--------|-------|-------------|
| `valid_i` | 1 | One-cycle pulse: new MoE layer starts |
| `n_experts_i` | 4 | Number of active experts (1–8) |
| `eid_i[0:7]` | 3×8 | Expert IDs, sorted **descending** by token count |
| `ntok_i[0:7]` | 7×8 | Token counts (descending) |
| `c2_cache_valid_i` | 1 | C2 SRAM has a valid cached expert |
| `c2_cache_eid_i` | 3 | Expert currently cached in C2 SRAM |
| `c3_cache_valid_i` | 1 | Same for C3 |
| `c3_cache_eid_i` | 3 | Same for C3 |
| `c2_done_i` | 1 | One-cycle pulse when C2 finishes an expert |
| `c3_done_i` | 1 | One-cycle pulse when C3 finishes an expert |

### Outputs (per-cluster assignment, one-cycle valid pulse)
| Signal | Width | Description |
|--------|-------|-------------|
| `c2_assign_o` | 1 | Assignment valid for C2 this cycle |
| `c2_eid_o` | 3 | Expert ID to assign to C2 |
| `c2_ntok_o` | 7 | Token count (may differ from original if SPLIT) |
| `c2_shape_s1_o` | 2 | SwishGLU DMA shape: 00=A, 01=B, 10=C |
| `c2_shape_s3_o` | 2 | DownProj DMA shape |
| `c2_cached_o` | 1 | 1 → skip Stage-1 DMA (cache hit) |
| `c3_*` | – | Mirror of C2 signals for cluster C3 |
| `ready_o` | 1 | High when idle, ready for next layer |
| `done_o` | 1 | High for one cycle when all experts dispatched & clusters idle |

---

## Shape Encoding

| Code | Name | M\_dim | BW req | T\_half (cc) |
|------|------|--------|--------|-------------|
| `2'b00` | ShapeA | 8 | 32 B/cc | 67 584 |
| `2'b01` | ShapeB | 4 | 64 B/cc | 33 792 |
| `2'b10` | ShapeC | 2 | 128 B/cc | 16 896 |

Physical BW constraint: xDMA + iDMA ≤ 128 B/cc total.

---

## Scheduling Policy

### Action Decision (combinational, 4 rules)

1. **SPLIT** – when `both_idle && tok[0] >= SPLIT_THR (8)` and either:
   - Only one expert remains, or
   - `tok[0] > 2 × tok[1]` (hot expert dominates)
2. **PAIR** – when `both_idle && n_active >= 2` (balanced distribution)
3. **SINGLE\_C2** – when `!both_idle && C2 free`
4. **SINGLE\_C3** – when `!both_idle && C3 free`

### Shape Selection (analytical)

| Scenario | S1 Shape | S3 Shape |
|----------|----------|----------|
| Cache hit | ShapeA (don't-care) | ShapeC |
| Concurrent (PAIR/SPLIT) | **ShapeB** (64+64=128 BW) | ShapeC |
| Solo (SINGLE) | ShapeC (128 BW, fastest) | ShapeC |
| Symmetric-small PAIR/SPLIT (both tok ≤ 4) | ShapeB | **ShapeB** |

The symmetric-small ShapeB for S3 prevents BW oversubscription when both
clusters start Stage-3 DMA simultaneously.

---

## FSM

```
  ┌──────────────────────────────────────────────────────┐
  │  IDLE  ──valid_i──►  ASSIGN  ──always──►  WAIT       │
  │    ▲                    ▲                  │ │       │
  │    │                    └──c2/c3_done──────┘ │       │
  │    │                       (!q_empty)         │       │
  │    │                                          │       │
  │    └──── DONE  ◄────── both_idle && q_empty ──┘       │
  │    └──── IDLE  ◄────── (!c2_busy && !c3_busy) ─────── │
  └──────────────────────────────────────────────────────┘
```

`done_o` pulses for **one cycle** when state = DONE and both clusters idle.
`ready_o` is high in state IDLE.

---

## Validation

The analytical shape rules were derived from beam-search analysis and achieve
≥ 92% match with the Python scheduler on training data.

Training data generation:  `Idea_Model/gen_hw_training.py`  
DT training & rule analysis: `Idea_Model/train_hw_policy.py`

---

## Integration Notes

- The Top-K router must **sort** expert (eid, ntok) pairs in descending
  token-count order before asserting `valid_i`.
- `valid_i` must be held for exactly **one cycle**.
- `c2_done_i` / `c3_done_i` must be one-cycle pulses; do not hold high.
- Cache state inputs (`c2_cache_valid_i`, `c2_cache_eid_i`) are latched on
  the `valid_i` cycle and override the scheduler's internal cache tracking.
- The scheduler **internally** updates its cache record when clusters report
  done, so it automatically tracks inter-layer cache state.
