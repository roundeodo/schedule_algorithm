from __future__ import annotations

import random
import unittest

from ..adapter import adapt_completed_candidate
from ..lowering import (
    StaticRunnerTemplate,
    decode_rtl_group,
    emit_rtl_group,
    lower_group,
    pack_rtl_record,
    unpack_rtl_record,
)
from ..model import (
    ContractConfig,
    DmaMask,
    DmaPolicy,
    ExpertSlice,
    GroupPlan,
    Phase,
    TICK_CC,
    build_streams,
    schedule_group,
    slices_from_counts,
)
from ..replay import replay_static_runner


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

    def test_rtl_descriptor_round_trip_preserves_complete_lists(self) -> None:
        plan = directed_plan()
        image = emit_rtl_group(plan)
        self.assertEqual(len(image.records), 8)
        self.assertEqual(decode_rtl_group(image), plan)
        for record in image.records:
            word = pack_rtl_record(record)
            self.assertLess(word.bit_length(), 26)
            self.assertEqual(unpack_rtl_record(word), record)

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

    def test_adapter_discards_planning_epoch_boundaries(self) -> None:
        plan = directed_plan()
        adapted = adapt_completed_candidate(plan.cluster0, plan.cluster1, group_id=7)
        self.assertEqual(adapted, plan)


if __name__ == "__main__":
    unittest.main()
