# Block-major N-outer execution contract

This isolated package is the Python/RTL-timespan/Bingo execution contract.  It
does not define a candidate generator or scorer.

## Correct execution order

The scheduler supplies two complete ordered expert-slice lists.  A static
worker expands each list as:

```text
phase -> weight block -> expert slice -> token tile
```

For `C0=[16]` and `C1=[4,2a,2b,2c,2d,2e,2f]`, every Gate/Up block executes:

```text
C0: 16
C1: 4 -> 2a -> 2b -> 2c -> 2d -> 2e -> 2f
```

Only after both streams advance does the worker return to the same expert
sequence for the next weight block.  Planning PAIR/SINGLE/SPLIT boundaries do
not exist in the execution stream.

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
97.96% aggregate DMA-lane utilization.  Host lowering and independent fixed-
runner replay reproduce every LOAD/COMPUTE timestamp exactly.

Run the focused tests from the Thesis root:

```bash
env PYTHONPATH=Idea_Model python3 -m unittest discover \
  -s Idea_Model/n_outer_scheduler/execution_contract/tests -t Idea_Model -v
```

## Files

- `model.py`: block-major stream, double buffering, DMA arbitration, makespan;
- `lowering.py`: complete group descriptor and fixed-runner command lowering;
- `replay.py`: independent dependency-only replay;
- `adapter.py`: adapts completed ordered lists only;
- `EXECUTION_CONTRACT.md`: frozen semantics and proof boundary;
- `BINGO_STATIC_RUNNER_SPEC.md`: static worker contract.
