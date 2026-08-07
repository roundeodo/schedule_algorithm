from __future__ import annotations

import random
import unittest

from ..adapter import adapt_completed_candidate
from ..lowering import (
    StaticRunnerTemplate,
    lower_group,
    lower_schedule_plan,
)
from ..model import (
    ContractConfig,
    DmaMask,
    DmaPolicy,
    ExpertSlice,
    GroupPlan,
    Phase,
    SchedulePlan,
    TICK_CC,
    build_schedule_streams,
    build_streams,
    schedule_group,
    schedule_plan,
    slices_from_counts,
)
from ..replay import replay_static_runner
from ..protocol import decode_scheduler_words, emit_scheduler_words
from ..runtime_interface import (
    NOuterWorkerArgs,
    RuntimeLayout,
    WorkerRole,
    decode_runtime_tables,
    lower_runtime_tables,
)


def directed_plan() -> GroupPlan:
    experts = slices_from_counts((16, 4, 2, 2, 2, 2, 2, 2))
    return GroupPlan((experts[0],), experts[1:], group_id=7)


class BlockMajorContractTest(unittest.TestCase):
    def test_default_tick_is_exact_lattice_quantum(self) -> None:
        self.assertEqual(TICK_CC, 1408)
        config = ContractConfig()
        self.assertEqual(config.gate_up.m4_compute_ticks, 4)
        self.assertEqual(config.gate_up.m2_compute_ticks, 2)
        self.assertEqual(config.down.m4_compute_ticks, 2)
        self.assertEqual(config.down.m2_compute_ticks, 1)

    def test_stream_order_is_phase_block_expert(self) -> None:
        plan = directed_plan()
        c0, c1 = build_streams(plan)
        self.assertEqual(
            [(item.phase, item.block_id, item.expert_slice.ntokens) for item in c0[:3]],
            [
                (Phase.GATE_UP, 0, 16),
                (Phase.GATE_UP, 1, 16),
                (Phase.GATE_UP, 2, 16),
            ],
        )
        self.assertEqual(
            [(item.phase, item.block_id, item.expert_slice.ntokens) for item in c1[:8]],
            [
                (Phase.GATE_UP, 0, 4),
                *[(Phase.GATE_UP, 0, 2)] * 6,
                (Phase.GATE_UP, 1, 4),
            ],
        )

    def test_directed_pipeline_fills_both_vcs_after_prime(self) -> None:
        result = schedule_group(directed_plan())
        self.assertTrue(result.validated)
        self.assertEqual(result.makespan_ticks, 196)
        self.assertEqual(result.lower_bound_ticks, 196)
        self.assertEqual(result.compute_lower_bound_ticks, 196)
        self.assertEqual(result.dma_lower_bound_ticks, 192)
        self.assertEqual(result.initial_wait_ticks, (4, 4))
        self.assertEqual(result.steady_stall_ticks, (0, 0))
        self.assertGreater(result.compute_utilization, 0.97)
        self.assertGreater(result.dma_lane_utilization, 0.97)

    def test_directed_next_expert_load_overlaps_current_compute(self) -> None:
        result = schedule_group(directed_plan())
        c1_compute = {
            event.item.key.stream_index: event
            for event in result.computes
            if event.item.key.cluster == 1
        }
        c1_load = {
            event.item.key.stream_index: event
            for event in result.loads
            if event.item.key.cluster == 1
        }
        # C(block0,E4) overlaps L(block0,E2a), then each E2 load/compute pair.
        self.assertEqual((c1_compute[0].start_tick, c1_compute[0].end_tick), (4, 8))
        self.assertEqual((c1_load[1].start_tick, c1_load[1].end_tick), (4, 8))
        for index in range(1, 6):
            self.assertEqual(c1_compute[index].start_tick, c1_load[index + 1].start_tick)
            self.assertEqual(c1_compute[index].end_tick, c1_load[index + 1].end_tick)

    def test_deadline_aware_arbiter_is_required_for_small_experts(self) -> None:
        plan = directed_plan()
        adaptive = schedule_group(plan)
        single = schedule_group(
            plan,
            config=ContractConfig(dma_policy=DmaPolicy.SINGLE_ONLY),
        )
        self.assertEqual(adaptive.makespan_ticks, 196)
        self.assertEqual(adaptive.steady_stall_ticks, (0, 0))
        self.assertEqual(single.makespan_ticks, 337)
        self.assertEqual(single.steady_stall_ticks, (0, 141))

    def test_initial_prime_splits_the_two_lanes(self) -> None:
        result = schedule_group(directed_plan())
        first = [event for event in result.loads if event.start_tick == 0]
        self.assertEqual(len(first), 2)
        self.assertEqual({event.lanes for event in first}, {DmaMask.IDMA, DmaMask.XDMA})

    def test_ping_pong_and_lane_resources_are_explicitly_legal(self) -> None:
        result = schedule_group(directed_plan())
        loads = {event.item.key: event for event in result.loads}
        computes = {event.item.key: event for event in result.computes}
        for cluster, stream in enumerate(result.streams):
            for index, item in enumerate(stream):
                self.assertLessEqual(loads[item.key].end_tick, computes[item.key].start_tick)
                if index >= 2:
                    self.assertLessEqual(
                        computes[stream[index - 2].key].end_tick,
                        loads[item.key].start_tick,
                    )

    def test_single_slot_compact_stream_preserves_complete_lists(self) -> None:
        plan = directed_plan()
        wrapped = SchedulePlan((plan,), schedule_id=plan.group_id)
        stream = emit_scheduler_words(wrapped)
        self.assertEqual(len(stream.words), 10)
        self.assertEqual(decode_scheduler_words(stream), wrapped)

    def test_lowered_runner_replays_every_tick_exactly(self) -> None:
        lowered = lower_group(directed_plan())
        replay = replay_static_runner(lowered.runner_program)
        self.assertEqual(replay.makespan_ticks, lowered.schedule.makespan_ticks)
        self.assertEqual(
            {event.key: (event.start_tick, event.end_tick) for event in replay.loads},
            {
                event.item.key: (event.start_tick, event.end_tick)
                for event in lowered.schedule.loads
            },
        )
        self.assertEqual(
            {event.key: (event.start_tick, event.end_tick) for event in replay.computes},
            {
                event.item.key: (event.start_tick, event.end_tick)
                for event in lowered.schedule.computes
            },
        )

    def test_fixed_runner_topology_does_not_depend_on_distribution(self) -> None:
        short = lower_group(GroupPlan((ExpertSlice(0, 0, 2),), ())).runner_program
        long = lower_group(directed_plan()).runner_program
        self.assertEqual(
            short.template.topology_signature,
            long.template.topology_signature,
        )
        self.assertEqual(short.template.topology_signature, StaticRunnerTemplate().topology_signature)

    def test_random_groups_lower_and_replay_exactly(self) -> None:
        rng = random.Random(0x4E4F5554)
        for _ in range(40):
            count = rng.randint(2, 10)
            experts = tuple(
                ExpertSlice(eid, 0, rng.randint(1, 24)) for eid in range(count)
            )
            split = rng.randint(1, count - 1)
            left = list(experts[:split])
            right = list(experts[split:])
            rng.shuffle(left)
            rng.shuffle(right)
            lowered = lower_group(GroupPlan(tuple(left), tuple(right)))
            replay = replay_static_runner(lowered.runner_program)
            self.assertEqual(replay.makespan_ticks, lowered.schedule.makespan_ticks)
            self.assertEqual(
                {event.key: (event.start_tick, event.end_tick) for event in replay.loads},
                {
                    event.item.key: (event.start_tick, event.end_tick)
                    for event in lowered.schedule.loads
                },
            )
            self.assertEqual(
                {event.key: (event.start_tick, event.end_tick) for event in replay.computes},
                {
                    event.item.key: (event.start_tick, event.end_tick)
                    for event in lowered.schedule.computes
                },
            )

    def test_split_slices_may_cross_clusters_but_not_overlap(self) -> None:
        plan = GroupPlan(
            (ExpertSlice(3, 0, 6),),
            (ExpertSlice(3, 6, 10),),
        )
        self.assertEqual(sum(item.ntokens for item in (*plan.cluster0, *plan.cluster1)), 16)
        with self.assertRaises(ValueError):
            GroupPlan(
                (ExpertSlice(3, 0, 8),),
                (ExpertSlice(3, 7, 9),),
            )

    def test_adapter_builds_one_completed_slot(self) -> None:
        plan = directed_plan()
        adapted = adapt_completed_candidate(plan.cluster0, plan.cluster1, group_id=7)
        self.assertEqual(adapted, plan)


class MultiSlotContractTest(unittest.TestCase):
    @staticmethod
    def plan() -> SchedulePlan:
        return SchedulePlan(
            (
                GroupPlan(
                    (ExpertSlice(0, 0, 2),),
                    (ExpertSlice(1, 0, 16),),
                    group_id=0,
                ),
                GroupPlan(
                    (ExpertSlice(2, 0, 2),),
                    (ExpertSlice(3, 0, 2),),
                    group_id=1,
                ),
            ),
            schedule_id=9,
        )

    def test_single_slot_wrapper_preserves_original_timing(self) -> None:
        slot = directed_plan()
        single = schedule_group(slot)
        wrapped = schedule_plan(SchedulePlan((slot,), schedule_id=99))
        self.assertEqual(single.makespan_ticks, wrapped.makespan_ticks)
        self.assertEqual(single.steady_stall_ticks, wrapped.steady_stall_ticks)
        self.assertEqual(
            [(event.start_tick, event.end_tick) for event in single.loads],
            [(event.start_tick, event.end_tick) for event in wrapped.loads],
        )
        self.assertEqual(
            [(event.start_tick, event.end_tick) for event in single.computes],
            [(event.start_tick, event.end_tick) for event in wrapped.computes],
        )

    def test_stream_order_is_local_slot_then_phase_block_slice(self) -> None:
        c0, c1 = build_schedule_streams(self.plan())
        self.assertEqual(
            (c0[15].slot_index, c0[15].phase, c0[15].block_id),
            (0, Phase.DOWN, 7),
        )
        self.assertEqual(
            (c0[16].slot_index, c0[16].phase, c0[16].block_id),
            (1, Phase.GATE_UP, 0),
        )
        self.assertEqual(
            (c1[15].slot_index, c1[15].phase, c1[15].block_id),
            (0, Phase.DOWN, 7),
        )
        self.assertEqual(
            (c1[16].slot_index, c1[16].phase, c1[16].block_id),
            (1, Phase.GATE_UP, 0),
        )

    def test_clusters_advance_slots_without_global_barrier(self) -> None:
        result = schedule_plan(self.plan())
        c0_slot1_start = min(
            event.start_tick
            for event in result.computes
            if event.item.key.cluster == 0 and event.item.slot_index == 1
        )
        c1_slot0_end = max(
            event.end_tick
            for event in result.computes
            if event.item.key.cluster == 1 and event.item.slot_index == 0
        )
        self.assertLess(c0_slot1_start, c1_slot0_end)
        self.assertEqual(c0_slot1_start, 30)
        self.assertEqual(c1_slot0_end, 198)

    def test_cross_slot_first_weight_prefetch_uses_continuous_buffers(self) -> None:
        result = schedule_plan(self.plan())
        c0_slot1_load = min(
            (event for event in result.loads
             if event.item.key.cluster == 0 and event.item.slot_index == 1),
            key=lambda event: event.start_tick,
        )
        c0_slot0_end = max(
            event.end_tick
            for event in result.computes
            if event.item.key.cluster == 0 and event.item.slot_index == 0
        )
        self.assertLess(c0_slot1_load.start_tick, c0_slot0_end)
        self.assertEqual((c0_slot1_load.start_tick, c0_slot1_load.end_tick), (28, 30))
        self.assertEqual(c0_slot0_end, 29)

    def test_real_token_intervals_cannot_overlap_across_slots(self) -> None:
        with self.assertRaises(ValueError):
            SchedulePlan(
                (
                    GroupPlan((ExpertSlice(5, 0, 8),), (), group_id=0),
                    GroupPlan((), (ExpertSlice(5, 7, 9),), group_id=1),
                )
            )

    def test_compact_rtl_stream_round_trip_preserves_all_slots(self) -> None:
        plan = self.plan()
        stream = emit_scheduler_words(plan)
        self.assertEqual(decode_scheduler_words(stream), plan)
        self.assertEqual(len(stream.words), 1 + 2 + 4)
        self.assertTrue(all(0 <= word < 1 << 64 for word in stream.words))

    def test_runtime_tables_round_trip_and_derive_slice_fields(self) -> None:
        plan = self.plan()
        tables = lower_runtime_tables(plan)
        self.assertEqual(decode_runtime_tables(tables), plan)
        self.assertEqual(tables.header.slot_count, 2)
        self.assertEqual(tables.header.total_slice_count, 4)
        first = tables.slices[0]
        self.assertEqual(first.token_ref_start, 0)
        self.assertEqual((first.m4_iters, first.m2_iters), (0, 1))
        hot = next(item for item in tables.slices if item.eid == 1)
        self.assertEqual(hot.token_ref_start, 256)
        self.assertEqual((hot.m4_iters, hot.m2_iters), (4, 0))

    def test_multislot_compact_lowering_replays_every_event(self) -> None:
        lowered = lower_schedule_plan(self.plan())
        replay = replay_static_runner(lowered.runner_program)
        self.assertEqual(replay.makespan_ticks, lowered.schedule.makespan_ticks)
        self.assertEqual(
            {event.key: (event.start_tick, event.end_tick) for event in replay.loads},
            {
                event.item.key: (event.start_tick, event.end_tick)
                for event in lowered.schedule.loads
            },
        )
        self.assertEqual(
            {event.key: (event.start_tick, event.end_tick) for event in replay.computes},
            {
                event.item.key: (event.start_tick, event.end_tick)
                for event in lowered.schedule.computes
            },
        )
        self.assertEqual(
            decode_scheduler_words(lowered.runner_program.scheduler_stream),
            self.plan(),
        )
        self.assertEqual(
            decode_runtime_tables(lowered.runner_program.runtime_tables),
            self.plan(),
        )

    def test_runtime_table_enforces_slot_workspace_capacity(self) -> None:
        with self.assertRaises(ValueError):
            lower_runtime_tables(
                self.plan(),
                layout=RuntimeLayout(slot_token_capacity=1),
            )

    def test_public_worker_args_are_schedule_level_not_block_level(self) -> None:
        args = NOuterWorkerArgs(
            schedule_header_addr=0x1000,
            static_context_addr=0x2000,
            runtime_sync_addr=0x3000,
            cluster_id=0,
            worker_role=WorkerRole.DMA_SLOT_WORKER,
        )
        self.assertEqual(args.cluster_id, 0)
        self.assertFalse(hasattr(args, "block_id"))
        self.assertFalse(hasattr(args, "dma_lane_mask"))

    def test_empty_slot_side_is_skipped_without_cross_cluster_wait(self) -> None:
        plan = SchedulePlan(
            (
                GroupPlan((ExpertSlice(0, 0, 2),), (), group_id=0),
                GroupPlan((), (ExpertSlice(1, 0, 2),), group_id=1),
            )
        )
        c0, c1 = build_schedule_streams(plan)
        self.assertTrue(all(item.slot_index == 0 for item in c0))
        self.assertTrue(all(item.slot_index == 1 for item in c1))
        lowered = lower_schedule_plan(plan)
        replay = replay_static_runner(lowered.runner_program)
        self.assertEqual(replay.makespan_ticks, lowered.schedule.makespan_ticks)

    def test_random_multislot_compact_runtime_and_replay_agree(self) -> None:
        rng = random.Random(0x534C4F54)
        for schedule_id in range(20):
            next_eid = 0
            slots = []
            for slot_id in range(rng.randint(2, 4)):
                clusters = []
                for _cluster in (0, 1):
                    items = []
                    for _ in range(rng.randint(0, 3)):
                        items.append(
                            ExpertSlice(next_eid, 0, rng.randint(1, 16))
                        )
                        next_eid += 1
                    clusters.append(tuple(items))
                if not clusters[0] and not clusters[1]:
                    clusters[rng.randrange(2)] = (
                        ExpertSlice(next_eid, 0, rng.randint(1, 16)),
                    )
                    next_eid += 1
                slots.append(GroupPlan(clusters[0], clusters[1], slot_id))
            plan = SchedulePlan(tuple(slots), schedule_id)
            lowered = lower_schedule_plan(plan)
            self.assertEqual(
                decode_scheduler_words(lowered.runner_program.scheduler_stream),
                plan,
            )
            self.assertEqual(
                decode_runtime_tables(lowered.runner_program.runtime_tables),
                plan,
            )
            self.assertEqual(
                replay_static_runner(lowered.runner_program).makespan_ticks,
                lowered.schedule.makespan_ticks,
            )


if __name__ == "__main__":
    unittest.main()
