# Frozen RTL scheduler policy contract

Policy ID: `r4-b2-k32-direct-v8-lpt-rem-snap-v1`

Status: frozen R4+bottom2/LPT v1 baseline contract.  It remains bit-exact for
reproducing the completed evaluation, but active P5 work in
`SCHEDULER_POLICY_SPEC.md` is evaluating an R8/K32 ordered-window ranker.  Do
not begin a new RTL implementation from this v1 contract until P5 either
passes its gates and publishes a replacement contract or is explicitly
rejected in favor of this baseline.

## Runtime boundary

One hardware invocation makes exactly one scheduling decision. It receives the
current remaining-expert state and the two complete cluster snapshots,
generates and scores at most 32 candidates, and returns one action. The state
transition is committed before the scheduler is invoked again. It does not
unroll the complete batch or future rounds.

Required state:

- remaining `(eid, ntok)` entries sorted by descending token count and then
  ascending expert ID;
- remaining-entry valid mask and count;
- C2 and C3 four-stage absolute timing boundaries;
- current/resident/prefetched expert IDs and full/S1-only residency flags;
- S1, S3, S2PF and S4PF DMA lane bindings and interval endpoints;
- the inherited admissible `f_score` pathmax bound.

All scheduler timing arithmetic is performed in `Tq = 11,264` cycle units.
The four-stage constants and every legal endpoint in the frozen model are
integer multiples of `Tq`. Conversion back to cycles occurs only at the
software-visible boundary.

The design dataset contains at most 64 experts, 256 tokens per expert and 512
total tokens. These are minimum supported limits, not permission to silently
saturate larger runtime inputs; configuration registers must reject unsupported
values.

## Candidate expert pool

For each decision, form the set union of:

1. remaining ranks R0 through R3;
2. the final two remaining ranks;
3. any remaining expert concretely named by a C2/C3 prefetch or residency.

Remove duplicate IDs. Timing-equivalent experts may be collapsed only when
their token count and all C2/C3 named-residency/cache observations are equal.
PAIR retains up to two IDs from one equivalence class because it may consume
two distinct experts.

## Candidate slots

The mode is `LAST_EXPERT` when one expert remains, `BOTH_IDLE` when the cluster
`task_end` values are equal, and `ONE_IDLE` otherwise. The first-pass family
quotas at K32 are:

| Mode | SINGLE | PAIR | SPLIT | PREFETCH |
| --- | ---: | ---: | ---: | ---: |
| BOTH_IDLE | 5 | 19 | 7 | 1 |
| ONE_IDLE | 26 | 1 | 4 | 1 |
| LAST_EXPERT | 30 | 0 | 1 | 1 |

If a family cannot fill its quota, unused slots are filled one at a time in
`SINGLE, PAIR, SPLIT, PREFETCH` order until K32 or exhaustion. Every emitted
physical action consumes one slot. Macro expansion is round-robin by micro
depth; it is not an uncounted local search.

The macro bank contains:

- SINGLE: legal selected expert on the earliest or concretely reserved cluster;
- PAIR: two distinct selected experts, with exact symmetric duplicates removed;
- SPLIT: one selected expert with `half +/- 1`, front/tail `1,2,4,8`, four
  future-release targets and one equal-finish cut;
- PREFETCH: selected expert into a legal empty S4PF slot.

Split target cuts use monotone binary crossing followed by a fixed +/-4 check.
They do not enumerate every token cut.

The micro-profile bank is `A/B`, `B/B`, `C/C`, with conditional `C/B`, `A/C`
and `B/C` forms when one stage is already cached. SINGLE uses eight fixed DMA
lane plans. PAIR/SPLIT use dedicated C2-iDMA and C3-xDMA lanes plus the fixed
opportunistic BOTH-lane form when the peer skips the same stage. S2PF patterns
are the fixed four combinations `00, 11, 01, 10`. Explicit S4PF uses the
earliest legal `dma3_end` start and one representative per resulting shape.

Candidate order is architectural state because it is the final tie-break. RTL
must reproduce the generator order in the golden model exactly.

## Exact child transition

Each candidate is rejected unless it satisfies all of the following:

- selected experts still exist and token counts are conserved;
- split halves sum to the selected expert's full token count;
- task start does not precede the prior cluster `task_end`;
- concrete reservation ownership is honored;
- cache-hit flags match the named residency at the candidate start;
- S2PF is used only for an uncached down stage;
- iDMA and xDMA intervals never overlap another user of the same lane;
- a non-prefetch action consumes at least one expert;
- explicit prefetch makes progress by creating a legal reservation.

The surviving action is applied with the exact four-stage timing equations.
The child pathmax is `max(parent.f_score, child_admissible_bound)`.

## Future score and selection

For each exact child:

```text
load2 = child.c2.task_end_q
load3 = child.c3.task_end_q

for each remaining expert in descending token rank:
    blocks = (ntok + 1) >> 1
    duration_q = blocks + (blocks << 1)
    if load2 <= load3:
        load2 += duration_q
    else:
        load3 += duration_q

lpt_q = max(child.f_score_q, load2, load3)
key = (lpt_q,
       child.remaining_count,
       max(child.c2.task_end_q, child.c3.task_end_q),
       candidate_index)
```

Commit the candidate with the lexicographically smallest key.

## Suggested iterative microarchitecture

The minimum-area implementation uses four blocks:

1. pool/macro generator;
2. micro-action instantiator plus legality and exact-transition unit;
3. candidate-local LPT scanner;
4. best-key register and action register.

The implementation may process one candidate at a time. With K32 and E64,
the LPT upper bound is 2,048 remaining-entry visits per decision. A one-entry
per-cycle scanner therefore completes scoring in at most 2,048 scan cycles,
plus bounded candidate construction and transition latency. This is below the
minimum 11,264-cycle timing quantum, but RTL timing closure must still be
measured rather than assumed. Parallel candidate scorers are optional and may
not change candidate order or arithmetic.

## RTL verification gates

- Python golden determinism: identical input produces identical action-history
  SHA-256 and makespan.
- Per-decision parity: candidate count, serialized candidate fields, score key
  and winning index match Python.
- All timing fields are exact in `Tq` units with no saturation or rounding.
- Candidate count never exceeds 32 and every nonterminal state emits at least
  one legal candidate.
- Complete RTL history passes the independent four-stage history validator.
- Regression covers E8, E32 and E64; all three modes; all four action families;
  cache-hit patterns; both DMA lanes; S2PF; S4PF; and split target rules.
