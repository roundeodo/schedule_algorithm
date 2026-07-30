#!/usr/bin/env python3
"""Construct replay-valid hot/cold block schedules for directed OLMoE cases.

This is an upper-bound constructor, not an independent optimality proof.  It
packs experts into synchronized blocks in units of the exact Shape-C token-pair
time.  One anchor expert occupies one cluster while an ordered companion list
occupies the other.  The first companion uses a disjoint single DMA lane; later
companions may use BOTH lanes after the anchor has prefetched its down weights.

Every abstract block is lowered through ``four_stage_scheduler``'s complete
action generator and the final history is replayed by the reference validator.
Equality with a previously certified lower bound is therefore a valid optimal
schedule certificate; a construction failure or a larger makespan proves
nothing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import four_stage_scheduler as reference
from run_four_stage_reference import serialize_action


TICK_CC = reference.SHAPE_C.T_s3
BLOCK_CC = reference.SHAPE_C.T_s1 + reference.SHAPE_C.T_s3


@dataclass(frozen=True)
class Expert:
    eid: int
    ntok: int

    @property
    def blocks(self) -> int:
        return (self.ntok + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM

    @property
    def first_companion_blocks(self) -> int:
        # An uncached 1/2-token task needs six ticks when it is restricted to
        # one DMA lane.  Tasks of at least three tokens already have enough
        # compute to hide the same single-lane transfer.
        return max(2, self.blocks)


@dataclass(frozen=True)
class Block:
    anchor: Expert
    companions: tuple[Expert, ...]
    first_cached: bool = False
    anchor_cached: bool = False

    @property
    def duration_blocks(self) -> int:
        return self.anchor.blocks

    @property
    def companion_blocks(self) -> int:
        if not self.companions:
            return 0
        first_blocks = (
            self.companions[0].blocks
            if self.first_cached
            else self.companions[0].first_companion_blocks
        )
        return first_blocks + sum(
            expert.blocks for expert in self.companions[1:]
        )


@dataclass(frozen=True)
class SplitBlock:
    expert: Expert
    left_tokens: int

    @property
    def right_tokens(self) -> int:
        return self.expert.ntok - self.left_tokens

    @property
    def left_blocks(self) -> int:
        return max(2, (self.left_tokens + 1) // 2)

    @property
    def right_blocks(self) -> int:
        return max(2, (self.right_tokens + 1) // 2)

    @property
    def duration_blocks(self) -> int:
        return max(self.left_blocks, self.right_blocks)


def _enumerate_companion_counts(
    remaining: tuple[int, ...], anchor_blocks: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every load-class multiset that fits beside one anchor."""
    subsets: list[tuple[int, ...]] = []
    counts = [0] * len(remaining)

    def visit(load_class: int, ordinary: int, has_long: bool) -> None:
        if load_class > anchor_blocks:
            subsets.append(tuple(counts))
            return
        for count in range(remaining[load_class] + 1):
            next_ordinary = ordinary + count * load_class
            next_has_long = has_long or (load_class >= 2 and count > 0)
            packed = (
                next_ordinary
                if next_has_long
                else next_ordinary + (1 if next_ordinary else 0)
            )
            if packed > anchor_blocks:
                break
            counts[load_class] = count
            visit(load_class + 1, next_ordinary, next_has_long)
        counts[load_class] = 0

    visit(1, 0, False)
    return tuple(subsets)


def _enumerate_companion_counts_with_credit(
    remaining: tuple[int, ...], anchor_blocks: int
) -> tuple[tuple[tuple[int, ...], bool], ...]:
    """Enumerate normal packs plus all-cold packs using one ready-S1 credit."""
    choices = [(counts, False) for counts in _enumerate_companion_counts(
        remaining, anchor_blocks
    )]
    maximum_cold = min(remaining[1], anchor_blocks)
    for cold_count in range(1, maximum_cold + 1):
        counts = [0] * len(remaining)
        counts[1] = cold_count
        choices.append((tuple(counts), True))
    return tuple(dict.fromkeys(choices))


@lru_cache(maxsize=None)
def _minimum_block_class_plan(
    state: tuple[int, ...],
) -> tuple[int, tuple[tuple[int, tuple[int, ...]], ...]]:
    """Exact DP for the bounded synchronized-block abstraction.

    The largest remaining load class must either anchor the next block or be a
    companion of an equal-size anchor.  Selecting one copy as the anchor and
    enumerating every fitting companion multiset is therefore complete for this
    abstraction.  The objective is total anchor blocks, i.e. final makespan in
    units of ``BLOCK_CC``.
    """
    if not any(state):
        return 0, ()
    anchor_blocks = max(index for index, count in enumerate(state) if count)
    remaining = list(state)
    remaining[anchor_blocks] -= 1
    remaining_tuple = tuple(remaining)
    best = None
    for companions in _enumerate_companion_counts(
        remaining_tuple, anchor_blocks
    ):
        next_state = tuple(
            remaining_tuple[index] - companions[index]
            for index in range(len(state))
        )
        tail_cost, tail_plan = _minimum_block_class_plan(next_state)
        total_cost = anchor_blocks + tail_cost
        ordinary = sum(
            index * companions[index] for index in range(len(companions))
        )
        has_long = any(companions[2:])
        packed = (
            ordinary
            if has_long
            else ordinary + (1 if ordinary else 0)
        )
        key = (total_cost, -packed, -sum(companions))
        candidate = (
            key,
            (
                total_cost,
                ((anchor_blocks, companions),) + tail_plan,
            ),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("block DP produced no transition")
    return best[1]


@lru_cache(maxsize=None)
def _minimum_block_class_plan_with_credit(
    state: tuple[int, ...], credits: int
) -> tuple[int, tuple[tuple[int, tuple[int, ...], bool], ...]]:
    """Exact block DP with a bounded number of already-prefetched cold starts."""
    if not any(state):
        return 0, ()
    anchor_blocks = max(index for index, count in enumerate(state) if count)
    remaining = list(state)
    remaining[anchor_blocks] -= 1
    remaining_tuple = tuple(remaining)
    choices = (
        _enumerate_companion_counts_with_credit(remaining_tuple, anchor_blocks)
        if credits > 0
        else tuple(
            (counts, False)
            for counts in _enumerate_companion_counts(
                remaining_tuple, anchor_blocks
            )
        )
    )
    best = None
    for companions, use_credit in choices:
        if use_credit and credits <= 0:
            continue
        next_state = tuple(
            remaining_tuple[index] - companions[index]
            for index in range(len(state))
        )
        tail_cost, tail_plan = _minimum_block_class_plan_with_credit(
            next_state, credits - int(use_credit)
        )
        total_cost = anchor_blocks + tail_cost
        packed = sum(
            index * companions[index] for index in range(len(companions))
        )
        if not use_credit and companions[1] and not any(companions[2:]):
            packed += 1
        key = (total_cost, -packed, -sum(companions), -int(use_credit))
        candidate = (
            key,
            (
                total_cost,
                ((anchor_blocks, companions, use_credit),) + tail_plan,
            ),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("credit block DP produced no transition")
    return best[1]


@lru_cache(maxsize=None)
def _minimum_producer_credit_plan(
    state: tuple[int, ...], ready_pair: bool
) -> tuple[
    int,
    tuple[tuple[int, tuple[int, ...], bool, bool], ...],
]:
    """Exact block DP with physically paired ready-S1 production.

    A zero-overhead cold-start credit consists of two resident S1 targets, one
    per cluster.  The DP uses an optimistic post-S2PF S4-window test to decide
    whether a synchronized pair might hold two non-overlapping S1 transfers;
    concrete start times and lanes are still decided by lowering.  The credit
    is valid only for the immediately next block because issuing any
    intervening task overwrites the resident slots.

    This DP is still an upper-bound constructor: concrete shapes, lane windows,
    and replay are checked during lowering.  It prevents the earlier optimistic
    error of treating one prefetched cold target as a complete credit.
    """
    if not any(state):
        return 0, ()
    anchor_blocks = max(index for index, count in enumerate(state) if count)
    remaining = list(state)
    remaining[anchor_blocks] -= 1
    remaining_tuple = tuple(remaining)
    choices = (
        _enumerate_companion_counts_with_credit(
            remaining_tuple, anchor_blocks
        )
        if ready_pair
        else tuple(
            (counts, False)
            for counts in _enumerate_companion_counts(
                remaining_tuple, anchor_blocks
            )
        )
    )
    best = None
    for companions, use_credit in choices:
        if use_credit and not ready_pair:
            continue
        next_state = tuple(
            remaining_tuple[index] - companions[index]
            for index in range(len(state))
        )
        companion_classes = [
            index
            for index, count in enumerate(companions)
            for _ in range(count)
        ]
        produces_credit = False
        if len(companion_classes) == 1:
            companion_blocks = companion_classes[0]
            # With S2PF, a q-block task exposes an optimistic q-tick S4
            # window.  Two next-S1 transfers can use BOTH for two ticks each.
            # Distinct q>=2 windows can be placed sequentially; equal q=2/3
            # windows are too short, while equal q>=4 can use separate single
            # lanes for four ticks.  Concrete lowering remains authoritative.
            produces_credit = (
                min(anchor_blocks, companion_blocks) >= 2
                and (
                    anchor_blocks != companion_blocks
                    or anchor_blocks >= 4
                )
            )
        tail_cost, tail_plan = _minimum_producer_credit_plan(
            next_state, produces_credit
        )
        total_cost = anchor_blocks + tail_cost
        ordinary = sum(
            index * companions[index]
            for index in range(len(companions))
        )
        has_long = any(companions[2:])
        packed = ordinary if has_long else ordinary + int(bool(ordinary))
        if use_credit:
            packed = ordinary
        key = (
            total_cost,
            -int(use_credit),
            -int(produces_credit),
            -packed,
            -sum(companions),
        )
        candidate = (
            key,
            (
                total_cost,
                (
                    (
                        anchor_blocks,
                        companions,
                        use_credit,
                        produces_credit,
                    ),
                )
                + tail_plan,
            ),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("producer-credit DP produced no transition")
    return best[1]


def build_producer_credit_block_plan(
    token_dist: dict[int, int], initial_ready_pair: bool = False
) -> tuple[Block, ...]:
    """Instantiate the paired-credit DP with concrete expert IDs."""
    experts = [
        Expert(eid, ntok) for eid, ntok in token_dist.items() if ntok > 0
    ]
    maximum = max((expert.blocks for expert in experts), default=0)
    histogram = [0] * (maximum + 1)
    by_class: dict[int, list[Expert]] = {}
    for expert in experts:
        histogram[expert.blocks] += 1
        by_class.setdefault(expert.blocks, []).append(expert)
    for members in by_class.values():
        members.sort(key=lambda expert: (-expert.ntok, expert.eid))

    _cost, class_plan = _minimum_producer_credit_plan(
        tuple(histogram), initial_ready_pair
    )
    blocks = []
    for anchor_class, companion_counts, use_credit, _produces in class_plan:
        anchor = by_class[anchor_class].pop(0)
        companions = []
        for load_class, count in enumerate(companion_counts):
            for _ in range(count):
                companions.append(by_class[load_class].pop(0))
        companions.sort(
            key=lambda expert: (
                expert.blocks < 2,
                -expert.blocks,
                -expert.ntok,
                expert.eid,
            )
        )
        block = Block(
            anchor,
            tuple(companions),
            first_cached=use_credit,
            anchor_cached=use_credit,
        )
        if block.companion_blocks > block.duration_blocks:
            raise AssertionError("producer-credit companion pack exceeds anchor")
        blocks.append(block)
    if any(by_class.values()):
        raise AssertionError("producer-credit DP left experts")
    return tuple(blocks)


def build_block_plan(
    token_dist: dict[int, int], cold_start_credits: int = 0
) -> tuple[Block, ...]:
    experts = [
        Expert(eid, ntok) for eid, ntok in token_dist.items() if ntok > 0
    ]
    maximum = max((expert.blocks for expert in experts), default=0)
    histogram = [0] * (maximum + 1)
    by_class: dict[int, list[Expert]] = {}
    for expert in experts:
        histogram[expert.blocks] += 1
        by_class.setdefault(expert.blocks, []).append(expert)
    for members in by_class.values():
        members.sort(key=lambda expert: (-expert.ntok, expert.eid))

    if cold_start_credits:
        _cost, credited_plan = _minimum_block_class_plan_with_credit(
            tuple(histogram), cold_start_credits
        )
        class_plan = credited_plan
    else:
        _cost, normal_plan = _minimum_block_class_plan(tuple(histogram))
        class_plan = tuple(
            (anchor_class, companion_counts, False)
            for anchor_class, companion_counts in normal_plan
        )
    blocks: list[Block] = []
    for anchor_class, companion_counts, first_cached in class_plan:
        anchor = by_class[anchor_class].pop(0)
        companions = []
        for load_class, count in enumerate(companion_counts):
            for _ in range(count):
                companions.append(by_class[load_class].pop(0))
        # Put a >=2-block task first when possible; this avoids the cold-task
        # single-lane startup penalty already represented by the DP.
        companions.sort(
            key=lambda expert: (
                expert.blocks < 2,
                -expert.blocks,
                -expert.ntok,
                expert.eid,
            )
        )
        # A cold first companion cannot overlap its BOTH-lane S3 with an
        # uncached anchor's foreground S1.  A zero-overhead credited block
        # therefore requires ready S1 weights on both clusters, not just on the
        # companion cluster.  Concrete lowering must materialize both slots.
        block = Block(
            anchor,
            tuple(companions),
            first_cached,
            anchor_cached=first_cached,
        )
        if block.companion_blocks > block.duration_blocks:
            raise AssertionError("companion pack exceeds anchor duration")
        blocks.append(block)
    if any(by_class.values()):
        raise AssertionError("block DP did not consume every expert")
    return tuple(blocks)


# ============================================================
#  Bounded-window constructive certificates
# ============================================================


def _load_blocks(ntok: int) -> int:
    return (int(ntok) + reference.FULL_M_DIM - 1) // reference.FULL_M_DIM


def _window_values(
    state: tuple[int, ...], top: int, bottom: int
) -> tuple[int, ...]:
    """Expanded load values visible in a TOP+BOTTOM ranked window."""
    ranked = tuple(
        ntok
        for ntok in range(len(state) - 1, 0, -1)
        for _ in range(state[ntok])
    )
    top_end = min(top, len(ranked))
    indices = list(range(top_end))
    if bottom:
        indices.extend(range(max(top_end, len(ranked) - bottom), len(ranked)))
    return tuple(ranked[index] for index in dict.fromkeys(indices))


def _remove_load(state: tuple[int, ...], ntok: int) -> tuple[int, ...]:
    if ntok <= 0 or ntok >= len(state) or state[ntok] <= 0:
        raise ValueError(f"cannot remove load {ntok} from {state}")
    updated = list(state)
    updated[ntok] -= 1
    return tuple(updated)


@lru_cache(maxsize=None)
def _window_tail_options(
    state: tuple[int, ...], capacity: int, top: int, bottom: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    """Every distinct later-companion packing reachable through the window.

    The first paired companion is handled separately because it must share one
    DMA lane and therefore costs at least two blocks.  Tasks issued after that
    pair may use BOTH lanes and cost their ordinary ``ceil(tokens/2)`` blocks.
    """
    choices: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {
        (state, 0): ()
    }
    if capacity <= 0:
        return tuple((end, seq, used) for (end, used), seq in choices.items())
    for ntok in dict.fromkeys(_window_values(state, top, bottom)):
        cost = _load_blocks(ntok)
        if cost > capacity:
            continue
        child = _remove_load(state, ntok)
        for end, tail, used in _window_tail_options(
            child, capacity - cost, top, bottom
        ):
            key = (end, cost + used)
            sequence = (ntok,) + tail
            previous = choices.get(key)
            if previous is None or sequence < previous:
                choices[key] = sequence
    return tuple(
        (end, sequence, used)
        for (end, used), sequence in choices.items()
    )


@lru_cache(maxsize=None)
def _window_block_transitions(
    state: tuple[int, ...],
    anchor_ntok: int,
    top: int,
    bottom: int,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, tuple[int, ...]], int], ...
]:
    """Enumerate one synchronized anchor block without hidden experts."""
    visible = _window_values(state, top, bottom)
    if anchor_ntok not in visible:
        return ()
    duration = _load_blocks(anchor_ntok)
    after_anchor = _remove_load(state, anchor_ntok)
    transitions: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, tuple[int, ...]], int],
    ] = {}

    # A solo anchor remains useful for unavoidable terminal imbalance.
    solo = (after_anchor, (anchor_ntok, ()), duration)
    transitions[(after_anchor, ())] = solo

    visible_counts = Counter(visible)
    for first in dict.fromkeys(visible):
        if first == anchor_ntok and visible_counts[first] < 2:
            continue
        first_cost = max(2, _load_blocks(first))
        if first_cost > duration:
            continue
        after_pair = _remove_load(after_anchor, first)
        for end, tail, used in _window_tail_options(
            after_pair, duration - first_cost, top, bottom
        ):
            companions = (first,) + tail
            key = (end, companions)
            transitions[key] = (
                end,
                (anchor_ntok, companions),
                first_cost + used,
            )
    return tuple(
        sorted(
            transitions.values(),
            key=lambda item: (-item[2], len(item[1][1]), item[1]),
        )
    )


@lru_cache(maxsize=None)
def _hot_anchor_window_tail(
    state: tuple[int, ...], budget: int, top: int, bottom: int
) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    """Find an exact-budget tail while always anchoring the hottest load.

    This is a constructive probe, not an infeasibility proof.  The one-block
    unlock enumeration below permits a non-hottest first anchor; subsequent
    hottest-anchor blocks keep the search bounded and match the intended RTL
    control structure.
    """
    useful_blocks = sum(
        state[ntok] * _load_blocks(ntok)
        for ntok in range(1, len(state))
    )
    if useful_blocks == 0:
        return () if budget == 0 else None
    if budget <= 0 or (useful_blocks + 1) // 2 > budget:
        return None
    anchor = max(ntok for ntok in range(1, len(state)) if state[ntok])
    duration = _load_blocks(anchor)
    if duration > budget:
        return None
    for child, block, _packed in _window_block_transitions(
        state, anchor, top, bottom
    ):
        tail = _hot_anchor_window_tail(child, budget - duration, top, bottom)
        if tail is not None:
            return (block,) + tail
    return None


def enumerate_window_block_class_plans(
    token_dist: dict[int, int],
    candidate_window: tuple[int, int],
    target_blocks: int,
    max_plans: int = 128,
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]:
    """Construct target-length plans with at most one initial unlock block.

    A returned plan is only an abstract candidate.  Concrete shape/binding
    lowering, per-action visibility, and full replay remain mandatory.
    """
    top, bottom = reference.normalize_candidate_window(candidate_window)
    maximum = max(token_dist.values(), default=0)
    histogram = [0] * (maximum + 1)
    for ntok in token_dist.values():
        if ntok > 0:
            histogram[int(ntok)] += 1
    initial = tuple(histogram)
    plans = []
    seen = set()

    direct = _hot_anchor_window_tail(initial, target_blocks, top, bottom)
    if direct is not None:
        plans.append(direct)
        seen.add(direct)

    # The initial block may intentionally consume a visible medium/cold class
    # so that the next middle-ranked expert enters TOP.  After this one unlock,
    # the bounded hottest-anchor tail is used again.
    for anchor in dict.fromkeys(_window_values(initial, top, bottom)):
        duration = _load_blocks(anchor)
        if duration > target_blocks:
            continue
        for child, first_block, _packed in _window_block_transitions(
            initial, anchor, top, bottom
        ):
            tail = _hot_anchor_window_tail(
                child, target_blocks - duration, top, bottom
            )
            if tail is None:
                continue
            plan = (first_block,) + tail
            if plan in seen:
                continue
            seen.add(plan)
            plans.append(plan)
            if len(plans) >= max_plans:
                return tuple(plans)
    return tuple(plans)


def instantiate_window_block_plan(
    token_dist: dict[int, int],
    class_plan: tuple[tuple[int, tuple[int, ...]], ...],
    candidate_window: tuple[int, int],
) -> tuple[Block, ...]:
    """Bind a load-class plan to concrete IDs while preserving visibility."""
    top, bottom = reference.normalize_candidate_window(candidate_window)
    remaining = tuple(sorted(token_dist.items(), key=lambda item: item[0]))

    def visible_entries() -> tuple[tuple[int, int], ...]:
        top_end = min(top, len(remaining))
        indices = list(range(top_end))
        if bottom:
            indices.extend(
                range(max(top_end, len(remaining) - bottom), len(remaining))
            )
        return tuple(remaining[index] for index in dict.fromkeys(indices))

    def choose(
        entries: tuple[tuple[int, int], ...],
        ntok: int,
        excluded: frozenset[int] = frozenset(),
    ) -> int:
        for eid, count in reversed(entries):
            if count == ntok and eid not in excluded:
                return eid
        raise RuntimeError(
            f"load {ntok} is hidden in concrete window {entries}"
        )

    def consume(eids: frozenset[int]) -> None:
        nonlocal remaining
        remaining = tuple(item for item in remaining if item[0] not in eids)

    blocks = []
    for anchor_ntok, companion_loads in class_plan:
        start_visible = visible_entries()
        anchor_eid = choose(start_visible, anchor_ntok)
        companions = []
        if companion_loads:
            first_ntok = companion_loads[0]
            first_eid = choose(
                start_visible, first_ntok, frozenset({anchor_eid})
            )
            companions.append(Expert(first_eid, first_ntok))
            consume(frozenset({anchor_eid, first_eid}))
            for ntok in companion_loads[1:]:
                eid = choose(visible_entries(), ntok)
                companions.append(Expert(eid, ntok))
                consume(frozenset({eid}))
        else:
            consume(frozenset({anchor_eid}))
        blocks.append(
            Block(Expert(anchor_eid, anchor_ntok), tuple(companions))
        )
    if remaining:
        raise RuntimeError(f"window class plan left {len(remaining)} experts")
    return tuple(blocks)


def enumerate_split_block_plans(
    token_dist: dict[int, int]
) -> tuple[tuple[SplitBlock | None, tuple[Block, ...], bool], ...]:
    """Enumerate no-split and one-SPLIT abstractions in cost order.

    A ready-S1 credit is not tied to SPLIT.  Any already issued task with a
    sufficiently long, conflict-free post-down window may legally prefetch the
    first cold companion of a later block.  Compare zero, one, and two credits
    explicitly; lowering below must still find and replay concrete producer
    actions, so an optimistic credited plan remains only an upper-bound probe.
    """
    candidates = []
    producer_blocks = build_producer_credit_block_plan(token_dist)
    producer_credits = sum(block.first_cached for block in producer_blocks)
    candidates.append(
        (
            (
                sum(block.duration_blocks for block in producer_blocks),
                0,
                producer_credits,
                -1,
                0,
            ),
            None,
            producer_blocks,
            bool(producer_credits),
        )
    )
    for credits in range(3):
        blocks = build_block_plan(token_dist, cold_start_credits=credits)
        used_credits = sum(block.first_cached for block in blocks)
        key = (
            sum(block.duration_blocks for block in blocks),
            0,
            used_credits,
            0,
            0,
        )
        candidates.append((key, None, blocks, bool(used_credits)))
    for eid, ntok in token_dist.items():
        if ntok < 2:
            continue
        remaining = dict(token_dist)
        del remaining[eid]
        tail_variants = []
        for credits in range(3):
            tail = build_block_plan(remaining, cold_start_credits=credits)
            tail_variants.append(
                (sum(block.first_cached for block in tail), tail)
            )
        expert = Expert(eid, ntok)
        for left_tokens in range(1, ntok):
            split = SplitBlock(expert, left_tokens)
            producer_tail = build_producer_credit_block_plan(
                remaining,
                initial_ready_pair=(
                    split.left_blocks >= 4 and split.right_blocks >= 4
                ),
            )
            producer_credits = sum(
                block.first_cached for block in producer_tail
            )
            candidates.append(
                (
                    (
                        split.duration_blocks
                        + sum(
                            block.duration_blocks
                            for block in producer_tail
                        ),
                        1,
                        producer_credits,
                        -1,
                        eid,
                    ),
                    split,
                    producer_tail,
                    bool(producer_credits),
                )
            )
            for used_credits, tail in tail_variants:
                total = split.duration_blocks + sum(
                    block.duration_blocks for block in tail
                )
                key = (
                    total,
                    1,
                    used_credits,
                    abs(split.left_tokens - split.right_tokens),
                    eid,
                )
                candidates.append(
                    (key, split, tail, bool(used_credits))
                )
    if not candidates:
        raise AssertionError("block planner produced no candidate")

    unique = []
    seen = set()
    for _key, split, blocks, has_credits in sorted(
        candidates, key=lambda item: item[0]
    ):
        signature = (
            None
            if split is None
            else (
                split.expert.ntok,
                min(split.left_tokens, split.right_tokens),
                max(split.left_tokens, split.right_tokens),
            ),
            tuple(
                (
                    block.anchor.ntok,
                    tuple(expert.ntok for expert in block.companions),
                    block.first_cached,
                    block.anchor_cached,
                )
                for block in blocks
            ),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((split, blocks, has_credits))
    return tuple(unique)


def build_split_block_plan(
    token_dist: dict[int, int]
) -> tuple[SplitBlock | None, tuple[Block, ...], bool]:
    """Return the cheapest abstract plan; lowering may try later ties."""
    return enumerate_split_block_plans(token_dist)[0]


def _candidate_remaining(experts: Iterable[Expert]) -> tuple[tuple[int, int], ...]:
    return tuple((expert.eid, expert.ntok) for expert in experts)


def _pair_action_candidates(
    state: reference.BeamState,
    anchor: Expert,
    first: Expert,
    start: int,
    require_anchor_s2pf: bool,
    first_cached: bool = False,
    anchor_cached: bool = False,
    first_cluster: int | None = None,
) -> tuple[reference.BeamState, ...]:
    expected_anchor_end = start + anchor.blocks * BLOCK_CC
    expected_first_end = start + (
        first.blocks if first_cached else first.first_companion_blocks
    ) * BLOCK_CC
    candidates = []
    for action in reference.gen_stage_actions(
        state.c2,
        state.c3,
        _candidate_remaining((anchor, first)),
    ):
        if action.c2_eid == anchor.eid and action.c3_eid == first.eid:
            anchor_cluster = 2
        elif action.c3_eid == anchor.eid and action.c2_eid == first.eid:
            anchor_cluster = 3
        else:
            continue
        if first_cluster is not None and 5 - anchor_cluster != first_cluster:
            continue
        if anchor_cached and not getattr(
            action, f"c{anchor_cluster}_s1_cached"
        ):
            continue
        if first_cached and not getattr(
            action, f"c{5 - anchor_cluster}_s1_cached"
        ):
            continue
        if require_anchor_s2pf and getattr(
            action, f"c{anchor_cluster}_s2pf_dma"
        ) == reference.DmaBinding.NONE:
            continue
        child = reference.apply_action(state, action)
        anchor_snap = getattr(child, f"c{anchor_cluster}")
        first_snap = getattr(child, f"c{5 - anchor_cluster}")
        if anchor_snap.task_start != start or first_snap.task_start != start:
            continue
        if (
            anchor_snap.task_end != expected_anchor_end
            or first_snap.task_end != expected_first_end
        ):
            continue
        candidates.append(
            (
                child.f_score,
                child.g_score,
                -int(
                    getattr(action, f"c{anchor_cluster}_s2pf_dma")
                    != reference.DmaBinding.NONE
                ),
                anchor_cluster,
                child,
            )
        )
    return tuple(
        item[-1] for item in sorted(candidates, key=lambda item: item[:-1])
    )


def _select_pair_action(
    state: reference.BeamState,
    anchor: Expert,
    first: Expert,
    start: int,
    require_anchor_s2pf: bool,
    first_cached: bool = False,
    anchor_cached: bool = False,
    first_cluster: int | None = None,
) -> reference.BeamState:
    candidates = _pair_action_candidates(
        state,
        anchor,
        first,
        start,
        require_anchor_s2pf,
        first_cached,
        anchor_cached,
        first_cluster,
    )
    if not candidates:
        raise RuntimeError(
            f"no pair lowering for E{anchor.eid}/{anchor.ntok} + "
            f"E{first.eid}/{first.ntok} at {start // TICK_CC} ticks"
        )
    return candidates[0]


def _select_single_action(
    state: reference.BeamState,
    expert: Expert,
    cluster: int | None,
    start: int,
    expected_end: int,
) -> reference.BeamState:
    candidates = []
    for action in reference.gen_stage_actions(
        state.c2,
        state.c3,
        _candidate_remaining((expert,)),
    ):
        selected_cluster = 2 if action.c2_eid == expert.eid else 3
        if getattr(action, f"c{5 - selected_cluster}_eid") >= 0:
            continue
        if cluster is not None and selected_cluster != cluster:
            continue
        child = reference.apply_action(state, action)
        snap = getattr(child, f"c{selected_cluster}")
        if snap.task_start != start or snap.task_end != expected_end:
            continue
        candidates.append((child.f_score, child.g_score, selected_cluster, child))
    if not candidates:
        raise RuntimeError(
            f"no single lowering for E{expert.eid}/{expert.ntok} at "
            f"{start // TICK_CC}->{expected_end // TICK_CC} ticks"
        )
    return min(candidates, key=lambda item: item[:-1])[-1]


def _split_action_candidates(
    state: reference.BeamState, split: SplitBlock, start: int
) -> tuple[reference.BeamState, ...]:
    expected = sorted(
        (
            start + split.left_blocks * BLOCK_CC,
            start + split.right_blocks * BLOCK_CC,
        )
    )
    candidates = []
    for action in reference.gen_stage_actions(
        state.c2,
        state.c3,
        _candidate_remaining((split.expert,)),
    ):
        if action.c2_eid != split.expert.eid or action.c3_eid != split.expert.eid:
            continue
        child = reference.apply_action(state, action)
        if child.c2.task_start != start or child.c3.task_start != start:
            continue
        if sorted((child.c2.task_end, child.c3.task_end)) != expected:
            continue
        candidates.append((child.f_score, child.g_score, child))
    return tuple(
        item[-1] for item in sorted(candidates, key=lambda item: item[:-1])
    )


def _select_split_action(
    state: reference.BeamState, split: SplitBlock, start: int
) -> reference.BeamState:
    candidates = _split_action_candidates(state, split, start)
    if not candidates:
        raise RuntimeError(
            f"no split lowering for E{split.expert.eid}/{split.expert.ntok} "
            f"duration class from cut {split.left_tokens}+{split.right_tokens} at "
            f"{start // TICK_CC} ticks"
        )
    return candidates[0]


def _select_split_prefetch_action(
    state: reference.BeamState,
    split: SplitBlock,
    target: Expert,
    start: int,
) -> tuple[reference.BeamState, int]:
    expected = sorted(
        (
            start + split.left_blocks * BLOCK_CC,
            start + split.right_blocks * BLOCK_CC,
        )
    )
    candidates = []
    for action in reference.gen_stage_actions(
        state.c2,
        state.c3,
        _candidate_remaining((split.expert,)),
    ):
        if action.c2_eid != split.expert.eid or action.c3_eid != split.expert.eid:
            continue
        child = reference.apply_action(state, action)
        if child.c2.task_start != start or child.c3.task_start != start:
            continue
        if sorted((child.c2.task_end, child.c3.task_end)) != expected:
            continue
        original_end = max(child.c2.task_end, child.c3.task_end)
        for pf_action in reference.gen_prefetch_actions(
            child.c2,
            child.c3,
            _candidate_remaining((target,)),
        ):
            if pf_action.pf_eid != target.eid:
                continue
            prefetched = reference.apply_action(child, pf_action)
            if max(prefetched.c2.task_end, prefetched.c3.task_end) != original_end:
                continue
            candidates.append(
                (
                    prefetched.f_score,
                    prefetched.g_score,
                    int(pf_action.pf_dma == reference.DmaBinding.BOTH),
                    pf_action.pf_cluster,
                    prefetched,
                )
            )
    if not candidates:
        raise RuntimeError(
            f"no free split-prefetch lowering for E{split.expert.eid}/"
            f"{split.expert.ntok} -> E{target.eid}/{target.ntok}"
        )
    selected = min(candidates, key=lambda item: item[:-1])
    return selected[-1], selected[-2]


def _free_prefetch_candidates(
    state: reference.BeamState,
    target: Expert,
) -> tuple[tuple[reference.BeamState, int], ...]:
    """Return concrete S4PF actions hidden by already committed work.

    The credited block abstraction assumes zero additional cluster occupancy,
    not merely an unchanged global maximum.  Therefore neither cluster's
    ``task_end`` may move.  Each returned state is a full explicit-lane action
    and remains subject to final history replay.
    """
    candidates = []
    seen = set()
    original_ends = (state.c2.task_end, state.c3.task_end)
    for action in reference.gen_prefetch_actions(
        state.c2,
        state.c3,
        _candidate_remaining((target,)),
    ):
        if action.pf_eid != target.eid:
            continue
        child = reference.apply_action(state, action)
        if (child.c2.task_end, child.c3.task_end) != original_ends:
            continue
        key = child.fingerprint()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((child, action.pf_cluster))
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item[0].f_score,
                item[0].g_score,
                item[1],
            ),
        )
    )


def _free_prefetch_pair_candidates(
    state: reference.BeamState,
    first: Expert,
    second: Expert,
) -> tuple[tuple[reference.BeamState, dict[int, int]], ...]:
    """Materialize two hidden S1 targets in the two distinct resident slots."""
    results = []
    seen = set()
    original_ends = (state.c2.task_end, state.c3.task_end)
    for left, right in ((first, second), (second, first)):
        for one, left_cluster in _free_prefetch_candidates(state, left):
            for two, right_cluster in _free_prefetch_candidates(one, right):
                if left_cluster == right_cluster:
                    continue
                if (two.c2.task_end, two.c3.task_end) != original_ends:
                    continue
                mapping = {
                    left.eid: left_cluster,
                    right.eid: right_cluster,
                }
                key = (two.fingerprint(), tuple(sorted(mapping.items())))
                if key in seen:
                    continue
                seen.add(key)
                results.append((two, mapping))
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item[0].f_score,
                item[0].g_score,
                tuple(sorted(item[1].items())),
            ),
        )
    )


def _lower_one_block(
    state: reference.BeamState,
    block: Block,
    prefetched_cluster: int | None = None,
) -> reference.BeamState:
    """Lower one synchronized abstract block from an immutable input state."""
    start = max(state.c2.task_end, state.c3.task_end)
    if not block.companions:
        if block.first_cached:
            raise RuntimeError("cached block has no first companion")
        expected_end = start + block.anchor.blocks * BLOCK_CC
        state = _select_single_action(
            state, block.anchor, None, start, expected_end
        )
    else:
        state = _select_pair_action(
            state,
            block.anchor,
            block.companions[0],
            start,
            require_anchor_s2pf=len(block.companions) > 1,
            first_cached=block.first_cached,
            anchor_cached=block.anchor_cached,
            first_cluster=(prefetched_cluster if block.first_cached else None),
        )
        anchor_cluster = 2 if state.c2.cur_eid == block.anchor.eid else 3
        companion_cluster = 5 - anchor_cluster
        for companion in block.companions[1:]:
            companion_start = getattr(state, f"c{companion_cluster}").task_end
            companion_end = companion_start + companion.blocks * BLOCK_CC
            state = _select_single_action(
                state,
                companion,
                companion_cluster,
                companion_start,
                companion_end,
            )

    actual_end = max(state.c2.task_end, state.c3.task_end)
    expected_block_end = start + block.duration_blocks * BLOCK_CC
    if actual_end != expected_block_end:
        raise RuntimeError(
            f"block E{block.anchor.eid} ended at {actual_end // TICK_CC}, "
            f"expected {expected_block_end // TICK_CC} ticks"
        )
    return state


def _lower_block_with_free_prefetch(
    state: reference.BeamState,
    block: Block,
    target: Expert,
) -> tuple[tuple[reference.BeamState, int], ...]:
    """Lower ``block`` while retaining one hidden next-S1 prefetch.

    A producer's pair action has multiple future-distinct DMA/S2PF variants.
    Selecting the locally smallest one before asking for S4PF can discard the
    only variant with a free lane window.  Enumerate those legal pair variants,
    attach the prefetch while the producer is still current, and only then
    lower later companions.  If later companions exist, the prefetch must be
    on the anchor cluster so their issue cannot overwrite its resident slot.
    """
    if block.first_cached:
        raise RuntimeError("a credited block cannot also produce its own credit")
    start = max(state.c2.task_end, state.c3.task_end)
    results = []
    seen = set()

    if not block.companions:
        produced = _lower_one_block(state, block)
        prefetched_variants = _free_prefetch_candidates(produced, target)
    else:
        pair_states = _pair_action_candidates(
            state,
            block.anchor,
            block.companions[0],
            start,
            require_anchor_s2pf=len(block.companions) > 1,
        )
        prefetched_variants = []
        for paired in pair_states:
            anchor_cluster = 2 if paired.c2.cur_eid == block.anchor.eid else 3
            companion_cluster = 5 - anchor_cluster
            for prefetched, cluster in _free_prefetch_candidates(paired, target):
                if len(block.companions) > 1 and cluster != anchor_cluster:
                    continue
                lowered = prefetched
                try:
                    for companion in block.companions[1:]:
                        companion_start = getattr(
                            lowered, f"c{companion_cluster}"
                        ).task_end
                        companion_end = (
                            companion_start + companion.blocks * BLOCK_CC
                        )
                        lowered = _select_single_action(
                            lowered,
                            companion,
                            companion_cluster,
                            companion_start,
                            companion_end,
                        )
                except RuntimeError:
                    continue
                prefetched_variants.append((lowered, cluster))

    expected_end = start + block.duration_blocks * BLOCK_CC
    for child, cluster in prefetched_variants:
        if max(child.c2.task_end, child.c3.task_end) != expected_end:
            continue
        key = child.fingerprint()
        if key in seen:
            continue
        seen.add(key)
        results.append((child, cluster))
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item[0].f_score,
                item[0].g_score,
                item[1],
            ),
        )
    )


def _lower_block_with_free_prefetch_pair(
    state: reference.BeamState,
    block: Block,
    first_target: Expert,
    second_target: Expert,
) -> tuple[tuple[reference.BeamState, dict[int, int]], ...]:
    """Lower one two-task producer and hide two next-S1 transfers."""
    if block.first_cached or len(block.companions) != 1:
        return ()
    start = max(state.c2.task_end, state.c3.task_end)
    results = []
    seen = set()
    for produced in _pair_action_candidates(
        state,
        block.anchor,
        block.companions[0],
        start,
        require_anchor_s2pf=False,
    ):
        for prefetched, mapping in _free_prefetch_pair_candidates(
            produced, first_target, second_target
        ):
            expected_end = start + block.duration_blocks * BLOCK_CC
            if max(prefetched.c2.task_end, prefetched.c3.task_end) != expected_end:
                continue
            key = (prefetched.fingerprint(), tuple(sorted(mapping.items())))
            if key in seen:
                continue
            seen.add(key)
            results.append((prefetched, mapping))
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item[0].f_score,
                item[0].g_score,
                tuple(sorted(item[1].items())),
            ),
        )
    )


def lower_block_plan(
    token_dist: dict[int, int],
    blocks: tuple[Block, ...],
    split: SplitBlock | None = None,
    split_prefetch: bool = False,
) -> reference.BeamState:
    scheduler = reference.FourStageScheduler(token_dist)
    initial = scheduler._initial_state()

    cached_blocks = tuple(block for block in blocks if block.first_cached)
    ordinary_blocks = tuple(block for block in blocks if not block.first_cached)
    if bool(cached_blocks) != split_prefetch:
        raise RuntimeError("credited-plan marker does not match cached blocks")

    # Credits are produced and consumed one at a time.  This respects the
    # one-entry resident slot on each cluster: no plan assumes that a prefetched
    # cold target survives an intervening task on the same cluster.  Backtrack
    # over producer order and explicit DMA-lane placement; the immutable state
    # objects make every failed branch side-effect free.
    def lower_credited_pairs(
        state: reference.BeamState,
        pending_cached: tuple[Block, ...],
        pending_ordinary: tuple[Block, ...],
    ) -> tuple[reference.BeamState, tuple[Block, ...]] | None:
        if not pending_cached:
            return state, pending_ordinary
        cached_block = pending_cached[0]
        if not cached_block.companions:
            raise RuntimeError("cached block has no companion target")
        first_target = cached_block.anchor
        second_target = cached_block.companions[0]
        for producer_index, producer in enumerate(pending_ordinary):
            for prefetched, mapping in _lower_block_with_free_prefetch_pair(
                state, producer, first_target, second_target
            ):
                first_cluster = mapping[second_target.eid]
                try:
                    consumed = _lower_one_block(
                        prefetched,
                        cached_block,
                        prefetched_cluster=first_cluster,
                    )
                except RuntimeError:
                    continue
                rest = (
                    pending_ordinary[:producer_index]
                    + pending_ordinary[producer_index + 1 :]
                )
                result = lower_credited_pairs(
                    consumed, pending_cached[1:], rest
                )
                if result is not None:
                    return result
        return None

    starting_points: list[
        tuple[reference.BeamState, tuple[Block, ...]]
    ] = []
    if split is None:
        starting_points.append((initial, cached_blocks))
    else:
        split_states = _split_action_candidates(initial, split, 0)
        if not split_states:
            raise RuntimeError(
                f"no split lowering for E{split.expert.eid}/"
                f"{split.expert.ntok}"
            )
        # Prefer using the SPLIT itself as the first two-slot producer.  This
        # is the legal pattern used by the previously proven 18 -> 9+9 case.
        if cached_blocks:
            cached_block = cached_blocks[0]
            first_target = cached_block.anchor
            second_target = cached_block.companions[0]
            for split_state in split_states:
                for prefetched, mapping in _free_prefetch_pair_candidates(
                    split_state, first_target, second_target
                ):
                    try:
                        consumed = _lower_one_block(
                            prefetched,
                            cached_block,
                            prefetched_cluster=mapping[second_target.eid],
                        )
                    except RuntimeError:
                        continue
                    starting_points.append((consumed, cached_blocks[1:]))
        starting_points.extend((state, cached_blocks) for state in split_states)

    credited_result = None
    for start_state, pending_cached in starting_points:
        credited_result = lower_credited_pairs(
            start_state, pending_cached, ordinary_blocks
        )
        if credited_result is not None:
            break
    if credited_result is None:
        raise RuntimeError(
            f"no legal inter-block producer for {len(cached_blocks)} "
            "ready-S1 credits"
        )
    state, remaining_blocks = credited_result
    for block in remaining_blocks:
        state = _lower_one_block(state, block)

    if state.remaining:
        raise RuntimeError(f"construction left {len(state.remaining)} experts")
    validated = reference.validate_schedule_history(state.history, token_dist)
    if validated != state.g_score:
        raise RuntimeError(
            f"constructed replay {validated} != state score {state.g_score}"
        )
    return state


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int) -> str:
    value = Fraction(cc, TICK_CC)
    return str(value.numerator) if value.denominator == 1 else str(value)


def _run_window_constructive(
    *,
    source_path: Path,
    proof_path: Path,
    output_path: Path,
    source: dict,
    prior: dict,
    names: list[str],
    candidate_window: tuple[int, int],
) -> int:
    """Emit replayed optimal witnesses for the bounded-window constructor."""
    window = reference.normalize_candidate_window(candidate_window)
    source_by_name = {row["name"]: row for row in source["cases"]}
    prior_by_name = {row["name"]: row for row in prior["cases"]}
    rows = []
    status_counts: Counter[str] = Counter()

    for index, name in enumerate(names, 1):
        case = source_by_name[name]
        prior_row = prior_by_name[name]
        counts = list(case.get("active_counts", prior_row["counts"]))
        if counts != list(prior_row["counts"]):
            raise RuntimeError(f"{name}: case input differs from frozen proof")
        token_dist = {
            eid: int(ntok)
            for eid, ntok in enumerate(counts)
            if int(ntok) > 0
        }
        target_ticks = Fraction(str(prior_row["best_reference_ticks"]))
        lower_ticks = Fraction(str(prior_row["certified_lower_bound_ticks"]))
        if not prior_row.get("proven_optimal") or target_ticks != lower_ticks:
            raise RuntimeError(f"{name}: prior target is not certified LB=UB")
        target_cc = target_ticks * TICK_CC
        if target_cc.denominator != 1 or int(target_cc) % BLOCK_CC:
            class_plans = ()
            failure = "certified target is not on the constructive block lattice"
        else:
            target_blocks = int(target_cc) // BLOCK_CC
            class_plans = enumerate_window_block_class_plans(
                token_dist, window, target_blocks
            )
            failure = "no bounded unlock/hot-anchor class plan at target"

        selected = None
        lowering_errors = []
        for plan_index, class_plan in enumerate(class_plans):
            try:
                blocks = instantiate_window_block_plan(
                    token_dist, class_plan, window
                )
                reference.clear_scheduler_caches()
                state = lower_block_plan(token_dist, blocks)
                if state.g_score != int(target_cc):
                    raise RuntimeError(
                        f"lowered {_ticks(state.g_score)} != target {target_ticks}"
                    )

                replay_state = reference.FourStageScheduler(
                    token_dist
                )._initial_state()
                for action_index, action in enumerate(state.history):
                    visible = reference.candidate_window_visible_eids(
                        replay_state.c2,
                        replay_state.c3,
                        replay_state.remaining,
                        window,
                    )
                    if not reference.action_within_candidate_window(
                        action, visible
                    ):
                        raise RuntimeError(
                            f"action {action_index} violates window {window}"
                        )
                    replay_state = reference.apply_action(replay_state, action)
                if replay_state.remaining:
                    raise RuntimeError("window replay is non-terminal")
                replay_cc = reference.validate_schedule_history(
                    state.history, token_dist
                )
                if replay_cc != int(target_cc):
                    raise RuntimeError(
                        f"independent replay {_ticks(replay_cc)} != {target_ticks}"
                    )
                selected = (plan_index, class_plan, blocks, state)
                break
            except RuntimeError as exc:
                lowering_errors.append(f"plan {plan_index}: {exc}")

        row = dict(prior_row)
        if selected is None:
            status = "construction_unresolved"
            # This payload is consumed as an optional witness source by the
            # window auditor.  Do not leave the prior unrestricted history
            # marked replay-valid here: it is globally valid but has not been
            # shown to obey this candidate window.
            row["actions"] = []
            row["history_replay_valid"] = False
            row["window_constructive_status"] = status
            row["window_constructive_error"] = (
                lowering_errors[-1] if lowering_errors else failure
            )
            row["window_constructive_plans_attempted"] = len(lowering_errors)
        else:
            plan_index, class_plan, blocks, state = selected
            status = "proved_sufficient"
            row.update(
                actions=[serialize_action(action) for action in state.history],
                best_reference_ticks=str(target_ticks),
                certified_lower_bound_ticks=str(target_ticks),
                optimality_gap=0.0,
                proven_optimal=True,
                history_replay_valid=True,
                termination=(
                    "restricted_window_constructive_history_equals_certified_lb"
                ),
                selected_history_source=(
                    "embedded_window_constructive_block_history"
                ),
                selected_proof_source=(
                    "embedded_window_constructive_block_history"
                ),
            )
            row["window_constructive_status"] = status
            row["window_constructive_evidence"] = {
                "candidate_window": list(window),
                "target_ticks": str(target_ticks),
                "history_replay_ticks": _ticks(state.g_score),
                "per_action_window_visible": True,
                "class_plan_index": plan_index,
                "class_plans_generated": len(class_plans),
                "actions": len(state.history),
                "construction_scope": (
                    "one optional window-unlock block followed by hottest-anchor blocks"
                ),
                "failure_is_not_infeasibility_proof": True,
            }
            row["window_constructive_class_plan"] = [
                {
                    "anchor_load": anchor,
                    "companion_loads": list(companions),
                }
                for anchor, companions in class_plan
            ]
            row["window_constructive_block_plan"] = [
                {
                    "anchor": [block.anchor.eid, block.anchor.ntok],
                    "companions": [
                        [expert.eid, expert.ntok]
                        for expert in block.companions
                    ],
                    "duration_ticks": _ticks(
                        block.duration_blocks * BLOCK_CC
                    ),
                }
                for block in blocks
            ]

        rows.append(row)
        status_counts[status] += 1
        print(
            f"[{index}/{len(names)}] {name} window={window} "
            f"plans={len(class_plans)} status={status}",
            flush=True,
        )

    script_path = Path(__file__).resolve()
    reference_path = Path(reference.__file__).resolve()
    payload = {
        "schema": "olmoe_window_constructive_block_witness_v1",
        "complete": True,
        "proof_model": "explicit_dma_lane_four_stage_bounded_window",
        "candidate_window": list(window),
        "summary": {
            "cases": len(rows),
            "proved_sufficient": status_counts["proved_sufficient"],
            "construction_unresolved": status_counts["construction_unresolved"],
            "all_histories_replay_valid": all(
                row.get("window_constructive_status") != "proved_sufficient"
                or row.get("history_replay_valid") is True
                for row in rows
            ),
            "all_actions_window_visible": all(
                row.get("window_constructive_status") != "proved_sufficient"
                or row.get("window_constructive_evidence", {}).get(
                    "per_action_window_visible"
                )
                is True
                for row in rows
            ),
        },
        "evidence": {
            "case_input": str(source_path.resolve()),
            "case_input_sha256": _sha256(source_path),
            "prior_proof": str(proof_path.resolve()),
            "prior_proof_sha256": _sha256(proof_path),
            "reference": str(reference_path),
            "reference_sha256": _sha256(reference_path),
            "script": str(script_path),
            "script_sha256": _sha256(script_path),
        },
        "cases": rows,
    }
    _atomic_write(output_path, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-input", type=Path, required=True)
    parser.add_argument("--prior-proof", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--candidate-window", nargs=2, type=int, metavar=("TOP", "BOTTOM")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.case_input.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_proof.read_text(encoding="utf-8"))
    source_by_name = {row["name"]: row for row in source["cases"]}
    prior_by_name = {row["name"]: row for row in prior["cases"]}
    names = [row["name"] for row in source["cases"]]
    if args.case:
        requested = set(args.case)
        missing = requested - set(names)
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
        names = [name for name in names if name in requested]

    if args.candidate_window is not None:
        return _run_window_constructive(
            source_path=args.case_input,
            proof_path=args.prior_proof,
            output_path=args.output,
            source=source,
            prior=prior,
            names=names,
            candidate_window=tuple(args.candidate_window),
        )

    rows = []
    status_counts: Counter[str] = Counter()
    for index, name in enumerate(names, 1):
        case = source_by_name[name]
        prior_row = prior_by_name[name]
        token_dist = {
            eid: ntok
            for eid, ntok in enumerate(case["active_counts"])
            if ntok > 0
        }
        plans = enumerate_split_block_plans(token_dist)
        split, blocks, split_prefetch = plans[0]
        abstract_cc = (
            (split.duration_blocks if split is not None else 0)
            + sum(block.duration_blocks for block in blocks)
        ) * BLOCK_CC
        prior_ub = Fraction(prior_row["best_reference_ticks"])
        abstract_ticks = Fraction(abstract_cc, TICK_CC)
        row = dict(prior_row)
        row["constructive_block_estimate_ticks"] = str(abstract_ticks)
        # Equal-cost SPLIT choices can expose different legal two-slot S4PF
        # windows.  Try every abstract plan that could improve the incumbent,
        # in nondecreasing estimated makespan order, until one fully lowers.
        lowering_errors = []
        selected = None
        for plan_index, (candidate_split, candidate_blocks, candidate_pf) in enumerate(
            plans
        ):
            candidate_cc = (
                (
                    candidate_split.duration_blocks
                    if candidate_split is not None
                    else 0
                )
                + sum(block.duration_blocks for block in candidate_blocks)
            ) * BLOCK_CC
            candidate_ticks = Fraction(candidate_cc, TICK_CC)
            if candidate_ticks >= prior_ub:
                break
            try:
                reference.clear_scheduler_caches()
                candidate_state = lower_block_plan(
                    token_dist,
                    candidate_blocks,
                    candidate_split,
                    candidate_pf,
                )
            except RuntimeError as exc:
                lowering_errors.append(
                    f"plan {plan_index} estimate={candidate_ticks}: {exc}"
                )
                continue
            selected = (
                candidate_split,
                candidate_blocks,
                candidate_pf,
                candidate_ticks,
                candidate_state,
                plan_index,
            )
            break

        try:
            if abstract_ticks >= prior_ub:
                status = "not_better_than_prior"
            elif selected is None:
                status = "lowering_failed"
                row["constructive_block_error"] = (
                    lowering_errors[-1]
                    if lowering_errors
                    else "no improving abstract plan"
                )
                row["constructive_plans_attempted"] = len(lowering_errors)
            else:
                (
                    split,
                    blocks,
                    split_prefetch,
                    abstract_ticks,
                    state,
                    selected_plan_index,
                ) = selected
                makespan = Fraction(state.g_score, TICK_CC)
                if makespan != abstract_ticks:
                    raise RuntimeError(
                        f"lowered makespan {makespan} != abstract {abstract_ticks}"
                    )
                lower_bound = Fraction(row["certified_lower_bound_ticks"])
                if makespan < lower_bound:
                    raise RuntimeError(
                        f"validated UB {makespan} below certified LB {lower_bound}"
                    )
                row.update(
                    best_reference_ticks=str(makespan),
                    optimality_gap=(
                        0.0
                        if makespan == lower_bound
                        else float((makespan - lower_bound) / lower_bound)
                    ),
                    proven_optimal=makespan == lower_bound,
                    termination=(
                        "constructive_block_history_equals_certified_lb"
                        if makespan == lower_bound
                        else "constructive_block_history_improved_upper_bound"
                    ),
                    history_replay_valid=True,
                    actions=[serialize_action(action) for action in state.history],
                    selected_history_source="embedded_constructive_block_history",
                    selected_proof_source="embedded_constructive_block_history",
                )
                row["constructive_block_plan"] = [
                    {
                        "anchor": [block.anchor.eid, block.anchor.ntok],
                        "companions": [
                            [expert.eid, expert.ntok]
                            for expert in block.companions
                        ],
                        "duration_ticks": _ticks(
                            block.duration_blocks * BLOCK_CC
                        ),
                        "first_cached": block.first_cached,
                        "anchor_cached": block.anchor_cached,
                    }
                    for block in blocks
                ]
                row["constructive_split"] = (
                    None
                    if split is None
                    else {
                        "expert": [split.expert.eid, split.expert.ntok],
                        "cut": [split.left_tokens, split.right_tokens],
                        "duration_ticks": _ticks(
                            split.duration_blocks * BLOCK_CC
                        ),
                        "prefetches_first_cold": split_prefetch,
                    }
                )
                row["constructive_selected_plan_index"] = selected_plan_index
                row["constructive_plans_attempted"] = selected_plan_index + 1
                status = "proved_optimal" if makespan == lower_bound else "improved_ub"
        except RuntimeError as exc:
            status = "lowering_failed"
            row["constructive_block_error"] = str(exc)
        row["constructive_block_status"] = status
        rows.append(row)
        status_counts[status] += 1
        print(
            f"[{index}/{len(names)}] {name} estimate={abstract_ticks} "
            f"prior={prior_ub} status={status}",
            flush=True,
        )

    summary = {
        "cases": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "proven_optimal": sum(bool(row["proven_optimal"]) for row in rows),
        "unproven": sum(not row["proven_optimal"] for row in rows),
    }
    _atomic_write(
        args.output,
        {
            "schema": "olmoe_constructive_block_schedules_v1",
            "complete": True,
            "summary": summary,
            "cases": rows,
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
