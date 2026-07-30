#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.candidates import (
    CandidateSkeleton,
    RemainingExpert,
    SliceAssignment,
    WindowSpec,
    canonical_shape,
    consume_candidate,
    generate_skeletons,
    materialize_modes,
    visible_experts,
    bounded_joint_mode_bank,
    rtl_symmetric_mode_bank,
)
from n_outer_scheduler.coarse_model.semantics import (
    ActionKind,
    DmaBinding,
    ExpertSlice,
    SHAPE_M2,
    SHAPE_M4,
    default_phases,
)


class CandidateTest(unittest.TestCase):

    def test_k4_always_starts_with_cluster_fixed_lane_baseline(self) -> None:
        expert = ExpertSlice(9, 0, 2)
        for cluster, expected in ((0, DmaBinding.IDMA), (1, DmaBinding.XDMA)):
            skeleton = CandidateSkeleton(
                ActionKind.SINGLE,
                (SliceAssignment(cluster, expert),),
            )
            bank = bounded_joint_mode_bank(skeleton, budget=4)
            self.assertEqual(bank[0].tasks[0].gate_up.dma, expected)
            self.assertEqual(bank[0].tasks[0].down.dma, expected)

    def test_symmetric_bank_has_only_fixed_and_all_both(self) -> None:
        skeleton = CandidateSkeleton(
            ActionKind.PAIR,
            (
                SliceAssignment(0, ExpertSlice(1, 0, 2)),
                SliceAssignment(1, ExpertSlice(2, 0, 2)),
            ),
        )
        bank = rtl_symmetric_mode_bank(skeleton)
        self.assertEqual(len(bank), 2)
        self.assertEqual(
            tuple(task.gate_up.dma for task in bank[0].tasks),
            (DmaBinding.IDMA, DmaBinding.XDMA),
        )
        self.assertTrue(
            all(
                phase.dma == DmaBinding.BOTH
                for task in bank[1].tasks
                for phase in (task.gate_up, task.down)
            )
        )

    def test_symmetric_bank_does_not_generate_dominated_both(self) -> None:
        skeleton = CandidateSkeleton(
            ActionKind.SINGLE,
            (SliceAssignment(0, ExpertSlice(1, 0, 8)),),
        )
        bank = rtl_symmetric_mode_bank(skeleton)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0].tasks[0].gate_up.dma, DmaBinding.IDMA)

    def test_top_bottom_union_is_stable_and_unique(self) -> None:
        remaining = tuple(RemainingExpert(eid, 0, count) for eid, count in enumerate((16, 8, 4, 2, 1)))
        visible = visible_experts(remaining, WindowSpec(2, 2))
        self.assertEqual([item.eid for item in visible], [0, 1, 4, 3])

    def test_three_token_split_never_fakes_two_plus_two(self) -> None:
        remaining = (RemainingExpert(5, 10, 3),)
        splits = [
            item
            for item in generate_skeletons(
                remaining, window=WindowSpec(1, 0), split_cuts="all"
            )
            if item.kind == ActionKind.SPLIT
        ]
        self.assertEqual(len(splits), 4)
        for split in splits:
            slices = split.assignments
            self.assertEqual(sum(item.expert_slice.ntokens for item in slices), 3)
            self.assertEqual(
                sorted(item.expert_slice.ntokens for item in slices), [1, 2]
            )
            self.assertEqual(consume_candidate(remaining, split), ())

    def test_incomplete_or_overlapping_external_split_is_rejected(self) -> None:
        remaining = (RemainingExpert(3, 0, 3),)
        incomplete = CandidateSkeleton(
            ActionKind.SPLIT,
            (
                SliceAssignment(0, ExpertSlice(3, 0, 1)),
                SliceAssignment(1, ExpertSlice(3, 2, 1)),
            ),
        )
        with self.assertRaises(ValueError):
            consume_candidate(remaining, incomplete)

    def test_canonical_shape_preserves_real_token_count(self) -> None:
        gate_up, _ = default_phases()
        self.assertEqual(canonical_shape(2, gate_up), SHAPE_M2)
        self.assertEqual(canonical_shape(3, gate_up), SHAPE_M4)

    def test_materialization_keeps_structure_and_adds_modes(self) -> None:
        skeleton = CandidateSkeleton(
            ActionKind.PAIR,
            (
                SliceAssignment(0, ExpertSlice(1, 0, 8)),
                SliceAssignment(1, ExpertSlice(2, 0, 2)),
            ),
        )
        modes = materialize_modes(skeleton)
        self.assertGreater(len(modes), 1)
        self.assertTrue(all(mode.kind == ActionKind.PAIR for mode in modes))
        self.assertTrue(all(mode.tasks[0].expert_slice.eid == 1 for mode in modes))


if __name__ == "__main__":
    unittest.main()
