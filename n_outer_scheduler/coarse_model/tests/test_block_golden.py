#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.block_golden import (
    build_block_streams,
    replay_best_policy,
    validate_block_result,
)
from n_outer_scheduler.coarse_model.candidates import (
    CandidateSkeleton,
    SliceAssignment,
)
from n_outer_scheduler.coarse_model.search import SelectedStep
from n_outer_scheduler.coarse_model.semantics import (
    ActionKind,
    DmaBinding,
    ExpertSlice,
    MacroActionPlan,
    MacroTaskPlan,
    PhasePlan,
    SHAPE_M8,
    evaluate_action,
)


def _single_step(eid: int, state=None) -> SelectedStep:
    expert_slice = ExpertSlice(eid, 0, 8)
    skeleton = CandidateSkeleton(
        ActionKind.SINGLE, (SliceAssignment(0, expert_slice),)
    )
    plan = MacroActionPlan(
        ActionKind.SINGLE,
        (
            MacroTaskPlan(
                0,
                expert_slice,
                PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                PhasePlan(SHAPE_M8, DmaBinding.IDMA),
            ),
        ),
    )
    timing = evaluate_action(plan, state=state) if state is not None else evaluate_action(plan)
    return SelectedStep(skeleton, plan, timing)


class BlockGoldenTest(unittest.TestCase):
    def test_stream_order_is_expert_then_phase_then_block(self) -> None:
        first = _single_step(3)
        second = _single_step(4, first.timing.next_state)
        stream, _ = build_block_streams((first, second))
        first_expert_items = [item for item in stream if item.eid == 3]
        self.assertEqual([item.phase_name for item in first_expert_items[:8]], ["gate_up"] * 8)
        self.assertEqual([item.phase_name for item in first_expert_items[8:]], ["down"] * 8)
        self.assertTrue(all(item.eid == 3 for item in stream[:16]))
        self.assertTrue(all(item.eid == 4 for item in stream[16:]))

    def test_next_expert_first_load_overlaps_previous_tail_compute(self) -> None:
        first = _single_step(3)
        second = _single_step(4, first.timing.next_state)
        result = replay_best_policy((first, second))
        validate_block_result(result)
        previous_tail = next(
            item for item in result.computes
            if item.item.eid == 3 and item.item.phase_name == "down" and item.item.block_id == 7
        )
        next_first = next(
            item for item in result.loads
            if item.item.eid == 4 and item.item.phase_name == "gate_up" and item.item.block_id == 0
        )
        self.assertLess(next_first.start_cc, previous_tail.end_cc)
        self.assertGreaterEqual(next_first.start_cc, previous_tail.start_cc)
        self.assertTrue(result.history_validated)


if __name__ == "__main__":
    unittest.main()

