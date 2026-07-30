#!/usr/bin/env python3

import unittest
from dataclasses import replace

from n_outer_scheduler.coarse_model.baselines import (
    paired_lpt_mode_search,
    split_hot_lpt_mode_search,
)
from n_outer_scheduler.coarse_model.bingo_task_abi import (
    BingoNOuterTaskKind,
    lower_history_to_bingo_tasks,
    replay_bingo_task_program,
)


class BingoTaskAbiTests(unittest.TestCase):
    def _check(self, distribution, result):
        program = lower_history_to_bingo_tasks(distribution, result.node)
        replay = replay_bingo_task_program(program)
        self.assertTrue(program.history_validated)
        self.assertEqual(replay.makespan_cc, program.source_block_makespan_cc)
        self.assertTrue(replay.resources_valid)
        self.assertTrue(replay.ping_pong_valid)
        self.assertTrue(replay.order_valid)
        self.assertTrue(replay.token_ranges_valid)
        self.assertEqual(len(program.tasks) % 2, 0)
        self.assertEqual(
            sum(task.kind == BingoNOuterTaskKind.LOAD_WEIGHT for task in program.tasks),
            sum(
                task.kind == BingoNOuterTaskKind.COMPUTE_RESIDENT_WEIGHT
                for task in program.tasks
            ),
        )

    def test_pair_history_lowers_to_dependency_complete_tasks(self):
        distribution = (16, 15, 6, 5, 4, 3, 2, 2)
        self._check(
            distribution,
            paired_lpt_mode_search(
                distribution, beam_width=1, mode_budget=4, score_mode="local"
            ),
        )

    def test_split_history_preserves_real_token_ranges(self):
        distribution = (31, 9, 6, 4, 2, 2)
        self._check(
            distribution,
            split_hot_lpt_mode_search(
                distribution,
                max_hot_experts=1,
                beam_width=1,
                mode_budget=4,
                score_mode="local",
            ),
        )

    def test_no_absolute_timestamp_dependency_is_required(self):
        distribution = (8, 7, 4, 2)
        result = paired_lpt_mode_search(
            distribution, beam_width=1, mode_budget=4, score_mode="local"
        )
        program = lower_history_to_bingo_tasks(distribution, result.node)
        shifted = type(program)(
            **{
                **program.__dict__,
                "tasks": tuple(
                    type(task)(
                        **{
                            **task.__dict__,
                            "model_start_cc": task.model_start_cc + 123,
                            "model_end_cc": task.model_end_cc + 123,
                        }
                    )
                    for task in program.tasks
                ),
            }
        )
        self.assertEqual(
            replay_bingo_task_program(shifted).makespan_cc,
            program.source_block_makespan_cc,
        )

    def test_task_token_range_cannot_diverge_from_macro_slot(self):
        distribution = (8, 7, 4, 2)
        result = paired_lpt_mode_search(
            distribution, beam_width=1, mode_budget=4, score_mode="local"
        )
        program = lower_history_to_bingo_tasks(distribution, result.node)
        tampered_task = replace(program.tasks[0], ntokens=program.tasks[0].ntokens + 1)
        tampered = replace(program, tasks=(tampered_task, *program.tasks[1:]))
        with self.assertRaises(AssertionError):
            replay_bingo_task_program(tampered)


if __name__ == "__main__":
    unittest.main()
