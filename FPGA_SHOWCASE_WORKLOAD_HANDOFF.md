# Four-policy scheduler showcase handoff

Status: Python-model-derived workload specification. All 16 case/method
histories pass four-stage replay, tick-lattice, token-slice, and explicit
IDMA/XDMA lane-overlap checks. The new four-policy workload set has not yet
been executed by the C translation, RTL, or FPGA.

## Source files

- Comparison evaluator and policy contract:
  `evaluate_scheduler_thesis_four_policy.py`
- Complete comparison result and action traces:
  `results/policy_search/scheduler_thesis_four_policy_showcases.json`
- Lowered per-cluster workload:
  `results/policy_search/scheduler_showcase_fpga_workloads.json`
- Flat per-task table:
  `results/policy_search/scheduler_showcase_fpga_tasks.csv`
- Reproducible lowering:
  `export_scheduler_showcase_fpga_workloads.py`

Run from `Idea_Model`:

```bash
python3 evaluate_scheduler_thesis_four_policy.py
python3 export_scheduler_showcase_fpga_workloads.py
```

The final Python policy is normative for `FULL_SCHEDULER`. The fixed-order
methods are independently replayable experiment plans; they are not outputs of
the final C or RTL scheduler.

## Four method definitions

| Method | Expert order | Physical parameters | Future scoring |
|---|---|---|---|
| `STATIC_DESC` | Global high-to-low queue; next free cluster refills | S1=B, S3=B; C2=iDMA, C3=xDMA; S2PF/S4PF off | None |
| `DYNAMIC_DESC` | Same global high-to-low queue | All legal shape/DMA/S2PF choices and concrete targeted S4PF | Current transition only |
| `DYNAMIC_TWO_ENDED` | C2 takes the hottest and C3 the coldest remaining expert; each refills independently | Exactly the same selector as `DYNAMIC_DESC` | Current transition only |
| `FULL_SCHEDULER` | Dynamically chooses PAIR/SINGLE/SPLIT, expert, cluster, and order | Compiled physical profiles with local reduction | Bounded continuation scorer |

The dynamic fixed-order selector minimizes, in order, current latest task end,
sum of task ends, task-end imbalance, and latest selected start. It does not
use beam search, rollout, remaining-work features, or the final continuation
scorer. It never splits an expert.

Shape B is the fixed baseline because its 64 B/cc requirement matches one DMA
lane per cluster and permits both clusters to load concurrently on disjoint
lanes. Shape A underuses the available lane bandwidth; Shape C requires both
lanes. Therefore B/B is an architecture-balanced fixed configuration, not a
case-specific weak setting.

## Selected distributions and results

One tick is 11,264 model clock cycles.

| Case | Compact distribution | Static desc. | Dynamic desc. | Dynamic two-ended | Full |
|---|---|---:|---:|---:|---:|
| certified OLMoE triple-hot | `22,18,14,3x19,2x8,1x13` | 162 | 159 | 137 | **129** |
| M70 three-hot/medium/cold | `28x3,6x4,2x16` | 132 | 126 | 127 | **105** |
| M92 parameter/order stress | `76,40,2x32,1x4` | 198 | 168 | 172 | **144** |
| M60 OLMoE-style high-skew/cold-tail stress | `36,22,13,6,2x17,1x9` | 138 | 133 | 111 | **99** |

The first full schedule equals the certified 129-tick optimum. The remaining
three are structured tests, not claims about the frequency of a real router
window. The M70 row has three hotspots at 28 assignments. With a uniform
64-expert, Top-2, 70-token baseline, the mean is 2.1875, so each hotspot is
12.8 times the mean. It also contains four medium experts and sixteen
two-token cold experts. Full scheduling is `126/105 = 1.200x` faster than
dynamic descending and `127/105 = 1.210x` faster than dynamic two-ended.

The M92 row is a structured stress case with 92 Top-2 input tokens, 38 active
experts, and 26 inactive experts. It isolates both intended benefits without
forcing every conceptual expert to be active:

- dynamic physical selection: `198/168 = 1.179x`;
- full scheduling after dynamic physical selection: `168/144 = 1.167x`;
- full scheduling versus dynamic two-ended: `172/144 = 1.194x`.

The M60 row is a structured OLMoE-style high-skew stress case, not a measured
router window. It has 120 assignments (60 Top-2 input tokens), four experts
above two assignments, seventeen two-assignment experts, and nine one-assignment
experts. Of 64 conceptual experts, 30 are active and 60 have load at most two.
Full scheduling is `138/99 = 1.394x` faster than static descending,
`133/99 = 1.343x` faster than dynamic descending, and `111/99 = 1.121x`
faster than dynamic two-ended.

The largest full-versus-dynamic-two-ended speedup among the selected rows is
`127/105 = 1.210x`. No representative distributed OLMoE-style case reaching
1.3x against dynamic two-ended was found, so the thesis must not claim such a
ratio for this four-policy comparison. The largest full-versus-static result
is now `138/99 = 1.394x`.

## Expert, token, and slot contract

`workload_eid` is a ranked ID: descending token count, with stable source-ID
tie breaking where a source ID exists. Experts omitted from the list have zero
tokens, but the conceptual layer still contains 64 experts.

The source datasets contain marginal expert counts rather than original
per-token Top-2 pairs. `token_routing` is a deterministic legal reconstruction
with two distinct experts per token and exactly the requested marginal count.
It must not be described as the original OLMoE router pairing.

`global_round` identifies one model decision and is not a runtime barrier.
`cluster_slot` is the normative local order. C2 and C3 execute their own slot
streams independently; one cluster may start its next slot before the other
finishes the previous global round.

For every task:

```text
token_ids = expert_loads[workload_eid].routed_token_ids[
    token_start_rank : token_start_rank + ntokens
]
```

This also defines the disjoint slices of a `SPLIT` action.

## Field mapping

| JSON field | Workload/C meaning |
|---|---|
| `cluster_abi_id` | `C2=0`, `C3=1` |
| `workload_eid` | expert ID used by the task |
| `token_start_rank`, `ntokens` | slice of this expert's routed tokens |
| `shape_s1.id`, `shape_s3.id` | `A=0`, `B=1`, `C=2` |
| `dma_s1.id`, `dma_s3.id`, `s2pf_dma.id` | `NONE=0`, `IDMA=1`, `XDMA=2`, `BOTH=3` |
| `skip_s1`, `skip_s3` | resident weight / prefetched weight hit |
| `has_s2pf` | down weight is transferred during S2 |
| `m_s2_exec`, `m_s4_exec` | number of M2 tail compute tiles |
| `s4pf` | concrete next-expert S1 prefetch attached to the prior same-cluster task |

The absolute `timing_ticks` fields are validation targets, not instructions to
busy-wait until a timestamp. The executable workload should express the data,
task, and DMA dependencies, record observed timestamps, and compare both
ordering and makespan with the model trace.

## Required implementation evidence

For every case and method, report:

1. input file hash, case name, and method name;
2. emitted tasks and per-cluster slot order;
3. emitted DMA operations and lane bindings;
4. task and DMA start/end timestamps;
5. preservation of each cluster-local stream;
6. absence of IDMA/XDMA overlap, with BOTH reserving both lanes;
7. consumer start only after required token and weight data are ready;
8. measured cycles and the tick conversion;
9. output-data correctness against the workload reference.

A successful program exit or matching final makespan alone is insufficient.
