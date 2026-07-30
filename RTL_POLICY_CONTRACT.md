# Archived P5 RTL scheduler policy contract

Policy ID: `r8-k32-direct-v9-full-lpt-dma-pm-rem-snap-p5`

Status: retained for reproduction and cost analysis only.  Full-LPT,
DMA-only pathmax and direct-v9 numeric ordering completed their historical
4,739-case validation, but the E64 local descriptor store and full-list scan
are not deployable under the confirmed slave-only, at-most-6+6-entry runtime
boundary.  Do not implement this file as the current RTL target.  The bounded
R4+bottom2/S2 policy in `SCHEDULER_POLICY_SPEC.md` passed closed-loop and
blind-v2 gates on 2026-07-19.  Its replacement RTL contract and
golden-equivalence harness are the next implementation deliverables.

## Runtime boundary

One hardware invocation makes exactly one scheduling decision. It receives the
current remaining-expert state and the two complete cluster snapshots,
generates and scores at most 32 candidates, and returns one action. The state
transition is committed before the scheduler is invoked again. It does not
unroll the complete batch or future rounds.

Required persistent state:

- up to 64 immutable `(eid, ntok)` descriptors sorted by descending token
  count and then ascending expert ID, written once by CVA6 at batch setup;
- one RTL-owned valid bit per descriptor and the remaining-entry count;
- C2 and C3 four-stage absolute timing boundaries;
- current/resident/prefetched expert IDs and full/S1-only residency flags;
- S1, S3, S2PF and S4PF DMA lane bindings and interval endpoints;
- the inherited admissible `f_score` pathmax bound.

RTL clears descriptor valid bits when actions commit.  It extracts the first
eight valid descriptors for candidate generation and scans all valid
descriptors for the future score.  CVA6 performs no per-round window refill,
bottom selection or shadow-window update.

Each descriptor keeps its immutable physical index `0..63`, equal to its
batch-initial sorted position.  Removing experts changes only the valid mask;
it never compacts or rewrites descriptors.  The first eight set bits are the
current R0..R7.  Candidate, cache and prefetch state carries the physical index
alongside the expert ID, so commit clears one or two exact bits without an EID
content-addressable search.  A sequential 64-bit valid-mask scan is sufficient
for top8 extraction and full-LPT traversal.

Exact four-stage task and DMA endpoints remain in the existing
`Tq = 11,264` cycle domain because every physical stage duration is an integer
multiple of `Tq`.  Candidate scoring uses a one-bit-wider
`Hq = 5,632` cycle domain: timeline endpoints are left-shifted by one on entry
to the scorer.  This represents the half-quantum result of sharing residual
DMA work across two lanes without widening the complete timeline datapath.
Conversion back to cycles occurs only at the software-visible boundary.

The design dataset contains at most 64 experts, 256 tokens per expert and 512
total tokens. These are minimum supported limits, not permission to silently
saturate larger runtime inputs; configuration registers must reject unsupported
values.

The stored task boundaries include `s1_end` even though later feasibility and
scoring mostly consume `dma1_end` and `s2_end`.  Direct-v9 uses complete
snapshot equality when equal `task_end` values decide whether C3 contributes a
second SINGLE macro stream; `s1_end` cannot in general be reconstructed from
the two surrounding endpoints without also retaining the chosen shape.

## Candidate expert pool

For each decision, form the set union of:

1. remaining ranks R0 through R7;
2. any remaining expert concretely named by a C2/C3 prefetch or residency.

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

Direct-v9 uses a fixed-width numeric physical-action key for its final local
tie-break.  Shapes are ordered `NONE,A,B,C`, lane masks are ordered
`NONE,iDMA,xDMA,BOTH`, and all IDs, token counts and Tq timestamps are compared
as integers.  The archived direct-v8 Python `repr()` ordering is retained only
for reproducing the old R4 policy and is not an RTL requirement.

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

The P5 transition stores a two-bit physical lane mask for every S1, S3, S2PF
and S4PF interval: `00=NONE`, `01=iDMA`, `10=xDMA`, `11=BOTH`.  A 64-B/cc
aggregate bandwidth field is insufficient because two simultaneous one-lane
transfers are legal only when their masks are disjoint.  Compute shape and DMA
binding remain independent; in particular Shape C may use one lane and Shape
A/B may use both lanes.  The baseline aggregate-BW timeline and checker are not
part of the P5 exact-transition path.

## Future score and selection

For each exact child, first form the mandatory DMA work.  Work is measured in
cycles on one 64-B/cc lane, or equivalently in `Hq` lane units:

```text
s1_slots   = distinct concrete remaining S1 reservations + valid ghost slots
full_slots = distinct concrete remaining full-residency reservations
s1_slots   = min(child.remaining_count, s1_slots)

dma_work_hq = 8 * (child.remaining_count - s1_slots)
            + 4 * (child.remaining_count - full_slots)
```

Starting at the earliest legal relaxed DMA release, sweep the committed S1,
S2PF, S3 and S4PF interval endpoints.  In each interval subtract
`interval_hq * free_lane_count` from `dma_work_hq`.  After the final committed
endpoint both lanes are free.  The first time the work reaches zero is
`child_dma_finish_hq`.

The only persistent future-bound register is:

```text
child_dma_pathmax_hq = max(parent_dma_pathmax_hq,
                           child.c2.task_end_hq,
                           child.c3.task_end_hq,
                           child_dma_finish_hq)
```

The compute, release-chain and critical-chain reference lower bounds are not
implemented.  The full-list LPT load already dominates them; the DMA-only
golden path reproduced the complete-bound history on all 4,739 validation
cases with zero history-hash mismatches.

Then scan the full remaining descriptor list:

```text
load2 = child.c2.task_end_hq
load3 = child.c3.task_end_hq

for each remaining expert in descending token rank:
    blocks = (ntok + 1) >> 1
    duration_hq = (blocks << 1) + (blocks << 2)
    if load2 <= load3:
        load2 += duration_hq
    else:
        load3 += duration_hq

lpt_hq = max(load2, load3)
score_hq = max(lpt_hq, child_dma_pathmax_hq)
key = (score_hq,
       child.remaining_count,
       max(child.c2.task_end_hq, child.c3.task_end_hq),
       candidate_index)
```

Commit the candidate with the lexicographically smallest key.  When it
commits, `child_dma_pathmax_hq` becomes the parent value for the next round.
No beam-search `f_score` or fitted coefficient is stored in RTL.

## Suggested iterative microarchitecture

The minimum-area implementation uses four blocks:

1. descriptor store, valid-mask rank extractor and pool/macro generator;
2. micro-action instantiator plus legality and exact-transition unit;
3. candidate-local LPT scanner;
4. best-key register and action register.

The best-action register stores the complete physical action and resulting
child state; P5 does not replay a narrow candidate ID through a second local
search.  This makes the architectural candidate index only a score tie-break
and prevents replay drift when a macro has multiple shape, S2PF, start or lane
variants.

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
- Timeline fields are exact in `Tq`; score and DMA-pathmax fields are exact in
  `Hq`, with no saturation or rounding.
- Candidate count never exceeds 32 and every nonterminal state emits at least
  one legal candidate.
- Complete RTL history passes the independent four-stage history validator.
- Regression covers E8, E32 and E64; all three modes; all four action families;
  cache-hit patterns; both DMA lanes; S2PF; S4PF; and split target rules.
