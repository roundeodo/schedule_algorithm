#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.semantics import (
    ActionKind,
    DmaBinding,
    ExpertSlice,
    LaneState,
    MacroActionPlan,
    MacroPhaseSpec,
    MacroScheduleState,
    MacroTaskPlan,
    PhasePlan,
    PrefetchTargetKind,
    SHAPE_M2,
    SHAPE_M4,
    SHAPE_M8,
    compute_block_cc,
    evaluate_action,
    evaluate_phase,
    legal_bindings,
    legal_prefetch_targets,
)


class MacroModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phase = MacroPhaseSpec("directed", 8, 128)

    def test_shape_coupled_binding_modes(self) -> None:
        self.assertEqual(
            legal_bindings(8, SHAPE_M8, self.phase),
            (DmaBinding.IDMA, DmaBinding.XDMA),
        )
        self.assertEqual(
            legal_bindings(4, SHAPE_M4, self.phase),
            (DmaBinding.IDMA, DmaBinding.XDMA),
        )
        self.assertEqual(
            legal_bindings(2, SHAPE_M2, self.phase),
            (DmaBinding.BOTH, DmaBinding.IDMA, DmaBinding.XDMA),
        )

    def test_m2_both_hides_while_single_stalls(self) -> None:
        both, _ = evaluate_phase(
            cluster=0,
            release_cc=0,
            ntokens=2,
            phase=self.phase,
            plan=PhasePlan(SHAPE_M2, DmaBinding.BOTH),
            lanes=LaneState(),
        )
        single, _ = evaluate_phase(
            cluster=0,
            release_cc=0,
            ntokens=2,
            phase=self.phase,
            plan=PhasePlan(SHAPE_M2, DmaBinding.IDMA),
            lanes=LaneState(),
        )
        self.assertEqual(both.end_cc, 9)
        self.assertEqual(both.pipeline_stall_cc, 0)
        self.assertEqual(single.end_cc, 17)
        self.assertEqual(single.pipeline_stall_cc, 7)

    def test_m8_both_is_rejected_as_dominated(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_phase(
                cluster=0,
                release_cc=0,
                ntokens=8,
                phase=self.phase,
                plan=PhasePlan(SHAPE_M8, DmaBinding.BOTH),
                lanes=LaneState(),
            )

    def test_pair_joint_service_never_overlaps_a_lane(self) -> None:
        phase = (self.phase, self.phase)
        action = MacroActionPlan(
            ActionKind.PAIR,
            (
                MacroTaskPlan(
                    0,
                    ExpertSlice(0, 0, 8),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                ),
                MacroTaskPlan(
                    1,
                    ExpertSlice(1, 0, 2),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                ),
            ),
        )
        result = evaluate_action(action, phases=phase)
        self.assertTrue(result.history_validated)
        self.assertGreaterEqual(result.makespan_cc, 1)
        self.assertGreaterEqual(result.resource_stall_cc, 0)

    def test_finite_joint_mode_bank_matches_exhaustive_for_m8_m2(self) -> None:
        action = MacroActionPlan(
            ActionKind.PAIR,
            (
                MacroTaskPlan(
                    0,
                    ExpertSlice(0, 0, 8),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                ),
                MacroTaskPlan(
                    1,
                    ExpertSlice(1, 0, 2),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                ),
            ),
        )
        bounded = evaluate_action(action, phases=(self.phase, self.phase))
        exhaustive = evaluate_action(
            action,
            phases=(self.phase, self.phase),
            exhaustive_service_orders=True,
        )
        self.assertEqual(bounded.makespan_cc, exhaustive.makespan_cc)

    def test_fixed_service_order_modes_are_legal_single_orders(self) -> None:
        action = MacroActionPlan(
            ActionKind.PAIR,
            (
                MacroTaskPlan(
                    0,
                    ExpertSlice(0, 0, 8),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                ),
                MacroTaskPlan(
                    1,
                    ExpertSlice(1, 0, 2),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                    PhasePlan(SHAPE_M2, DmaBinding.BOTH),
                ),
            ),
        )
        for mode in ("fixed_c0", "binding_hot", "binding_chain"):
            timing = evaluate_action(
                action,
                phases=(self.phase, self.phase),
                service_order_mode=mode,
            )
            self.assertTrue(timing.history_validated)
            self.assertEqual(len(timing.service_order), 8)

    def test_next_expert_first_block_is_charged_but_hidden_in_tail(self) -> None:
        task = lambda eid: MacroTaskPlan(
            0,
            ExpertSlice(eid, 0, 8),
            PhasePlan(SHAPE_M8, DmaBinding.IDMA),
            PhasePlan(SHAPE_M8, DmaBinding.IDMA),
        )
        first = evaluate_action(
            MacroActionPlan(ActionKind.SINGLE, (task(0),)),
            phases=(self.phase, self.phase),
        )
        second = evaluate_action(
            MacroActionPlan(ActionKind.SINGLE, (task(1),)),
            state=first.next_state,
            phases=(self.phase, self.phase),
        )
        gate = second.task_timings[0].gate_up
        self.assertEqual(
            gate.first_prefetch_release_cc,
            first.task_timings[0].down.last_compute_start_cc,
        )
        self.assertLess(gate.first_dma_end_cc, first.next_state.cluster_free_cc[0])
        self.assertEqual(gate.fill_stall_cc, 0)

    def test_first_blocks_are_real_dma_work_not_boolean_metadata(self) -> None:
        action = MacroActionPlan(
            ActionKind.SINGLE,
            (
                MacroTaskPlan(
                    0,
                    ExpertSlice(0, 0, 8),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                    PhasePlan(SHAPE_M8, DmaBinding.IDMA),
                ),
            ),
        )
        result = evaluate_action(action, phases=(self.phase, self.phase))
        first = [item for item in result.dma_intervals if item.role == "first_block"]
        internal = [item for item in result.dma_intervals if item.role == "internal_blocks"]
        self.assertEqual(len(first), 2)
        self.assertEqual(len(internal), 0)
        self.assertEqual(len(result.dma_intervals), 2 * self.phase.block_count)
        transferred = sum(
            (item.end_cc - item.start_cc)
            * (1 if item.binding in (DmaBinding.IDMA, DmaBinding.XDMA) else 2)
            * 64
            for item in result.dma_intervals
        )
        self.assertEqual(transferred, 2 * self.phase.block_count * self.phase.weight_block_bytes)

    def test_split_requires_disjoint_slices_of_same_expert(self) -> None:
        plan = lambda cluster, start, count: MacroTaskPlan(
            cluster,
            ExpertSlice(3, start, count),
            PhasePlan(SHAPE_M4, DmaBinding.IDMA if cluster == 0 else DmaBinding.XDMA),
            PhasePlan(SHAPE_M4, DmaBinding.IDMA if cluster == 0 else DmaBinding.XDMA),
        )
        action = MacroActionPlan(ActionKind.SPLIT, (plan(0, 0, 4), plan(1, 4, 4)))
        self.assertEqual(action.tasks[0].expert_slice.token_end, 4)
        with self.assertRaises(ValueError):
            MacroActionPlan(ActionKind.SPLIT, (plan(0, 0, 5), plan(1, 4, 4)))

    def test_two_buffers_force_internal_successor_until_phase_tail(self) -> None:
        current = ExpertSlice(7, 0, 4)
        down = MacroPhaseSpec("down", 8, 64)
        internal = legal_prefetch_targets(
            current_slice=current,
            phase=self.phase,
            block_id=3,
        )
        self.assertEqual(len(internal), 1)
        self.assertEqual(
            internal[0].kind, PrefetchTargetKind.SAME_EXPERT_NEXT_BLOCK
        )
        self.assertEqual(internal[0].block_id, 4)

        gate_up = MacroPhaseSpec("gate_up", 8, 128)
        boundary = legal_prefetch_targets(
            current_slice=current,
            phase=gate_up,
            block_id=7,
            down_phase=down,
        )
        self.assertEqual(
            boundary[0].kind, PrefetchTargetKind.SAME_EXPERT_DOWN_FIRST
        )

    def test_down_tail_can_only_prime_the_next_scheduled_slice(self) -> None:
        current = ExpertSlice(7, 0, 4)
        next_slice = ExpertSlice(9, 0, 2)
        down = MacroPhaseSpec("down", 8, 64)
        targets = legal_prefetch_targets(
            current_slice=current,
            phase=down,
            block_id=7,
            next_slice=next_slice,
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(
            targets[0].kind, PrefetchTargetKind.NEXT_EXPERT_GATE_UP_FIRST
        )
        self.assertEqual(targets[0].eid, 9)


if __name__ == "__main__":
    unittest.main()
