# Block-major N-outer execution contract

This isolated package is the Python/RTL-timespan/Bingo execution contract.  It
does not define a candidate generator or scorer.

## Correct execution order

The scheduler supplies ordered slots, each with a C0 and C1 expert-slice list.
Every cluster independently expands its non-empty local slots as:

```text
local slot -> phase -> weight block -> expert slice -> token tile
```

For `C0=[16]` and `C1=[4,2a,2b,2c,2d,2e,2f]`, every Gate/Up block executes:

```text
C0: 16
C1: 4 -> 2a -> 2b -> 2c -> 2d -> 2e -> 2f
```

Within a slot, the worker completes the expert sequence before returning to
the same sequence for the next weight block.  There is no global slot barrier:
a faster cluster may enter its next slot while the other cluster remains in
the prior slot.

## Pipeline

Each cluster has a LOAD worker, a COMPUTE worker, and two ping/pong weight
buffers.  The load producer may be one item ahead:

```text
Compute(item[k]) || Load(item[k+1])
```

The two global DMA lanes are explicit.  The deadline-aware arbiter chooses two
parallel single-lane transfers or one BOTH transfer.  A request that cannot be
served waits and the resulting VersaCore stall increases makespan.

The default time unit is one scheduler tick (`1408` accelerator cycles).  All
public timing fields use integer ticks.

## Directed acceptance case

For `C0=[16]`, `C1=[4,2,2,2,2,2,2]`:

```text
deadline-aware: makespan=196 ticks, initial_wait=(4,4), steady_stall=(0,0)
single-only:    makespan=337 ticks, initial_wait=(4,4), steady_stall=(0,141)
```

The cold-model lower bound is also 196 ticks: each cluster has 192 ticks of
compute after a mandatory four-tick initial load.  The directed schedule is
therefore optimal under this frozen resource model, not merely stall-free.

The deadline-aware result has 97.96% aggregate VersaCore utilization and
97.96% aggregate DMA-lane utilization.  Compact scheduler-word lowering,
runtime-table lowering, and independent fixed-runner replay reproduce every
audit LOAD/COMPUTE timestamp exactly.

Run the focused tests from the Thesis root:

```bash
env PYTHONPATH=Idea_Model python3 -m unittest discover \
  -s Idea_Model/n_outer_scheduler/execution_contract/tests -t Idea_Model -v
```

## Files

- `model.py`: multi-slot block-major stream, double buffering, DMA arbitration;
- `protocol.py`: compact 64-bit schedule/slot/slice RTL output records;
- `runtime_interface.py`: CVA6-derived schedule/slot/slice runtime tables;
- `lowering.py`: compact-interface and verification-only command lowering;
- `replay.py`: independent dependency-only replay;
- `adapter.py`: adapts completed ordered lists only;
- `EXECUTION_CONTRACT.md`: frozen semantics and proof boundary;
- `INTERFACE_PROTOCOL.md`: exact 64-bit fields and schedule-level worker ABI;
- `BINGO_STATIC_RUNNER_SPEC.md`: static worker contract.
- `MULTI_SLOT_INTERFACE_PROPOSAL.md`: design rationale and interface boundary.
