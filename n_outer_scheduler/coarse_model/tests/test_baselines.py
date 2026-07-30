#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.baselines import (
    fixed_lane_lpt,
    paired_lpt_mode_search,
    split_hot_lpt_mode_search,
)
from n_outer_scheduler.coarse_model.block_golden import replay_best_policy
from n_outer_scheduler.coarse_model.semantics import DmaBinding


class BaselineTest(unittest.TestCase):
    def test_fixed_lane_lpt_covers_each_expert_once(self) -> None:
        result = fixed_lane_lpt((16, 4, 2, 2))
        self.assertTrue(result.history_validated)
        self.assertEqual(
            sorted((*result.cluster_eids[0], *result.cluster_eids[1])),
            [0, 1, 2, 3],
        )
        golden = replay_best_policy(result.node.history)
        self.assertTrue(golden.history_validated)
        self.assertLessEqual(golden.makespan_cc, result.node.makespan_cc)

    def test_paired_lpt_search_replays(self) -> None:
        result = paired_lpt_mode_search((16, 4, 2, 2), beam_width=4)
        self.assertTrue(result.history_validated)
        self.assertFalse(result.node.remaining)

    def test_symmetric_two_entry_mode_bank_replays(self) -> None:
        result = paired_lpt_mode_search(
            (16, 15, 6, 5, 4, 3, 2, 2),
            beam_width=1,
            score_mode="local",
            service_order_mode="binding_chain",
            tie_break_mode="bank_order",
            pair_mode_policy="no_mixed",
            mode_bank_policy="rtl_symmetric2",
        )
        self.assertTrue(result.history_validated)
        self.assertFalse(result.node.remaining)
        self.assertTrue(result.name.endswith("rtl_symmetric2"))

    def test_both_on_tie_changes_only_the_equal_score_choice(self) -> None:
        fixed = paired_lpt_mode_search(
            (2, 2),
            beam_width=1,
            score_mode="local",
            service_order_mode="binding_chain",
            tie_break_mode="bank_order",
            pair_mode_policy="no_mixed",
            mode_bank_policy="rtl_symmetric2",
        )
        both = paired_lpt_mode_search(
            (2, 2),
            beam_width=1,
            score_mode="local",
            service_order_mode="binding_chain",
            tie_break_mode="both_on_tie",
            pair_mode_policy="no_mixed",
            mode_bank_policy="rtl_symmetric2",
        )
        self.assertEqual(fixed.node.makespan_cc, both.node.makespan_cc)
        self.assertEqual(
            fixed.node.history[0].plan.tasks[0].gate_up.dma,
            DmaBinding.IDMA,
        )
        self.assertTrue(
            all(
                task.gate_up.dma == DmaBinding.BOTH
                for task in both.node.history[0].plan.tasks
            )
        )

    def test_forced_split_preserves_original_token_coverage(self) -> None:
        result = split_hot_lpt_mode_search(
            (16, 4, 2, 2), max_hot_experts=1, beam_width=1
        )
        self.assertTrue(result.history_validated)
        self.assertEqual(result.node.history[0].plan.kind.value, "split")


if __name__ == "__main__":
    unittest.main()
