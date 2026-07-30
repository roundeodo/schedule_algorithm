#!/usr/bin/env python3

import unittest
import math

from n_outer_scheduler.model import (
    ExpertDescriptor,
    GroupDescriptor,
    NOuterSimulator,
    ScheduleCandidate,
    default_config,
    NOuterConfig,
    PhaseSpec,
)
from n_outer_scheduler.reference import ExactDmaPlanner
from n_outer_scheduler.search import generate_partition_candidates, search_candidates
from n_outer_scheduler.scheduler import (
    NOuterScheduler,
    SchedulerMode,
    SchedulerOptions,
)
from n_outer_scheduler.task_stream import (
    StartupMode,
    TaskKind,
    lower_schedule_to_tasks,
    replay_task_stream,
)
from n_outer_scheduler.compare_four_stage import _work_signature


class NOuterModelTest(unittest.TestCase):
    def test_token_lowering_covers_every_token_without_capacity_loss(self) -> None:
        expected = {
            1: (0, 1, 0, 1),
            2: (0, 1, 0, 2),
            3: (1, 0, 3, 0),
            4: (1, 0, 0, 0),
            5: (1, 1, 0, 1),
            6: (1, 1, 0, 2),
            7: (2, 0, 3, 0),
            8: (2, 0, 0, 0),
        }
        for ntokens, lowered in expected.items():
            expert = ExpertDescriptor(0, ntokens)
            actual = (
                expert.m4_iters,
                expert.m2_iters,
                expert.m4_tail_valid_tokens,
                expert.m2_valid_tokens,
            )
            self.assertEqual(actual, lowered)
            self.assertGreaterEqual(4 * expert.m4_iters + 2 * expert.m2_iters, ntokens)

    def test_static_stream_uses_dynamic_counts(self) -> None:
        simulator = NOuterSimulator(default_config())
        group = GroupDescriptor(
            cluster0=(ExpertDescriptor(0, 16),),
            cluster1=(ExpertDescriptor(1, 4), ExpertDescriptor(2, 2)),
        )
        streams = simulator.build_streams(group)
        total_blocks = sum(phase.block_count for phase in simulator.config.phases)
        self.assertEqual(len(streams[0]), total_blocks)
        self.assertEqual(len(streams[1]), 2 * total_blocks)
        self.assertEqual(streams[1][0].expert.eid, 1)
        self.assertEqual(streams[1][1].expert.eid, 2)
        self.assertEqual(streams[1][2].expert.eid, 1)

    def test_example_history_is_valid_and_starts_with_split_lanes(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((16, 4, 2, 2, 2, 2, 2, 2))
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=(experts[0],), cluster1=experts[1:]),
            label="directed-example",
        )
        result = NOuterSimulator(default_config()).evaluate(candidate)
        self.assertTrue(result.history_validated)
        first = [record for record in result.loads if record.start_cc == 0]
        self.assertEqual(len(first), 2)
        self.assertEqual({record.lanes for record in first}, {(0,), (1,)})

    def test_search_returns_valid_candidate(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((16, 4, 2, 2))
        )
        simulator = NOuterSimulator(default_config())
        result = search_candidates(
            simulator,
            generate_partition_candidates(experts),
            top_k=5,
        )
        self.assertTrue(result.best.history_validated)
        self.assertGreaterEqual(result.best.makespan_cc, result.best.lower_bound_cc)

    def test_small_exhaustive_order_mode_covers_all_ordered_partitions(self) -> None:
        experts = tuple(ExpertDescriptor(eid, eid + 1) for eid in range(4))
        candidates = generate_partition_candidates(
            experts, exhaustive_permutation_limit=4
        )
        # With the largest expert pinned to C0, the number of ordered
        # partitions is n! * (n + 1) / 2.
        self.assertEqual(len(candidates), math.factorial(4) * 5 // 2)
        self.assertEqual(len({candidate.label for candidate in candidates}), 60)

    def test_capacity_filter_rejects_only_invalid_candidates(self) -> None:
        config = default_config()
        constrained = type(config)(
            phases=config.phases,
            dma_policy=config.dma_policy,
            force_initial_split=config.force_initial_split,
            max_group_tokens_per_cluster=8,
        )
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((5, 3, 2, 2))
        )
        result = search_candidates(
            NOuterSimulator(constrained),
            generate_partition_candidates(experts),
            top_k=3,
        )
        self.assertGreater(result.rejected_candidates, 0)
        self.assertTrue(result.best.history_validated)
        for cluster in (0, 1):
            self.assertLessEqual(
                sum(
                    expert.ntokens
                    for expert in result.best.candidate.group.experts(cluster)
                ),
                8,
            )

    def test_exact_dma_search_can_improve_the_fast_policy(self) -> None:
        config = NOuterConfig(
            phases=(
                PhaseSpec(
                    name="directed",
                    block_count=3,
                    weight_block_bytes=128,
                    m4_compute_cc=1,
                    m2_compute_cc=1,
                ),
            )
        )
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((6, 4, 2))
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=experts[:2], cluster1=experts[2:]),
            label="exact-gap-directed",
        )
        exact = ExactDmaPlanner(NOuterSimulator(config)).evaluate(candidate)
        self.assertEqual(exact.heuristic_makespan_cc, 12)
        self.assertEqual(exact.makespan_cc, 11)
        self.assertTrue(exact.proven_optimal)
        self.assertTrue(exact.plan_validated)

    def test_full_size_directed_schedule_is_dma_optimal(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((16, 4, 2, 2, 2, 2, 2, 2))
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=(experts[0],), cluster1=experts[1:]),
            label="full-size-directed",
        )
        exact = ExactDmaPlanner(NOuterSimulator(default_config())).evaluate(candidate)
        self.assertEqual(exact.makespan_cc, 275968)
        self.assertEqual(exact.heuristic_gap_cc, 0)
        self.assertEqual(exact.decisions[0].grants, ((0, 0b01), (1, 0b10)))
        self.assertTrue(exact.proven_optimal)
        self.assertTrue(exact.plan_validated)

    def test_fast_scheduler_selects_directed_mapping_and_audits_prefetch(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((16, 4, 2, 2, 2, 2, 2, 2))
        )
        result = NOuterScheduler(
            options=SchedulerOptions(mode=SchedulerMode.FAST)
        ).schedule(experts)
        self.assertEqual([expert.ntokens for expert in result.candidate.group.cluster0], [16])
        self.assertEqual(
            [expert.ntokens for expert in result.candidate.group.cluster1],
            [4, 2, 2, 2, 2, 2, 2],
        )
        self.assertTrue(result.bandwidth.valid)
        self.assertEqual(result.bandwidth.peak_bandwidth_bytes_per_cc, 128)
        self.assertGreater(result.prefetch.overlapped_loads, 0)
        self.assertGreater(result.prefetch.phase_boundary_prefetches, 0)

    def test_reference_scheduler_proves_small_candidate_bank(self) -> None:
        config = NOuterConfig(
            phases=(PhaseSpec("small", 2, 256, 5, 3),)
        )
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((4, 2, 2))
        )
        result = NOuterScheduler(
            NOuterSimulator(config),
            SchedulerOptions(mode=SchedulerMode.REFERENCE),
        ).schedule(experts)
        self.assertTrue(result.dma_optimal_for_selected_candidate)
        self.assertTrue(result.optimal_within_generated_bank)
        self.assertTrue(result.bandwidth.valid)
        self.assertTrue(result.schedule.history_validated)

    def test_cold_task_stream_exactly_replays_model_history(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((8, 4, 2, 2))
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=experts[:1], cluster1=experts[1:]),
            label="task-stream-cold",
        )
        schedule = NOuterSimulator(default_config()).evaluate(candidate)
        stream = lower_schedule_to_tasks(schedule, startup_mode=StartupMode.COLD)
        replay = replay_task_stream(stream)

        self.assertEqual(replay.makespan_cc, schedule.makespan_cc)
        self.assertTrue(replay.dependencies_valid)
        self.assertTrue(replay.resources_valid)
        self.assertEqual(len(stream.issue_order), len(stream.tasks))
        self.assertTrue(any(task.dma_lane_mask == 3 for task in stream.tasks))

    def test_preloaded_task_stream_omits_only_first_load_per_cluster(self) -> None:
        experts = tuple(
            ExpertDescriptor(eid, ntokens)
            for eid, ntokens in enumerate((8, 4, 2))
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=experts[:1], cluster1=experts[1:]),
            label="task-stream-preloaded",
        )
        schedule = NOuterSimulator(default_config()).evaluate(candidate)
        cold = lower_schedule_to_tasks(schedule, startup_mode=StartupMode.COLD)
        preloaded = lower_schedule_to_tasks(
            schedule, startup_mode=StartupMode.PRELOADED_FIRST
        )
        replay = replay_task_stream(preloaded)

        self.assertEqual(len(preloaded.tasks), len(cold.tasks) - 2)
        first_computes = [
            task
            for task in preloaded.tasks
            if task.kind == TaskKind.COMPUTE_BLOCK and task.stream_index == 0
        ]
        self.assertEqual(len(first_computes), 2)
        self.assertTrue(all(task.weight_preloaded for task in first_computes))
        self.assertTrue(replay.resources_valid)
        self.assertLess(replay.makespan_cc, cold.source_makespan_cc)

    def test_task_compute_arguments_cover_scheduler_descriptor(self) -> None:
        expert = ExpertDescriptor(
            eid=7,
            ntokens=6,
            token_ref_start=23,
            split_token_start=2,
        )
        candidate = ScheduleCandidate(
            GroupDescriptor(cluster0=(expert,), cluster1=()),
            label="task-args",
        )
        schedule = NOuterSimulator(default_config()).evaluate(candidate)
        stream = lower_schedule_to_tasks(schedule)
        compute = next(
            task for task in stream.tasks if task.kind == TaskKind.COMPUTE_BLOCK
        )
        self.assertEqual(compute.eid, 7)
        self.assertEqual(compute.ntokens, 6)
        self.assertEqual(compute.token_ref_start, 23)
        self.assertEqual(compute.split_token_start, 2)
        self.assertEqual((compute.m4_iters, compute.m2_iters), (1, 1))

    def test_nouter_and_four_stage_account_for_identical_atomic_work(self) -> None:
        for ntokens in range(1, 33):
            signature = _work_signature((ntokens,))
            self.assertGreater(signature["compute_cc"], 0)
            self.assertGreater(signature["weight_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
