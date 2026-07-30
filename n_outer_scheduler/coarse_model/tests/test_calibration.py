#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.baselines import paired_lpt_mode_search
from n_outer_scheduler.coarse_model.calibration import (
    calibrate_history_mode_choices,
    calibrate_root_candidates,
)
from n_outer_scheduler.coarse_model.candidates import WindowSpec


class CalibrationTest(unittest.TestCase):
    def test_calibration_reports_timing_error_and_ranking_regret(self) -> None:
        result = calibrate_root_candidates(
            (8, 2), window=WindowSpec(2, 0), max_plans=12
        )
        self.assertEqual(len(result.entries), 12)
        self.assertGreaterEqual(result.max_abs_timing_error_cc, 0)
        self.assertGreaterEqual(result.max_mode_ranking_regret_cc, 0)

    def test_history_calibration_holds_prefix_and_skeleton_fixed(self) -> None:
        distribution = (9, 7, 4, 2)
        result = paired_lpt_mode_search(
            distribution,
            beam_width=1,
            mode_budget=4,
            score_mode="local",
        )
        calibration = calibrate_history_mode_choices(
            distribution, result.node, mode_budget=4
        )
        self.assertEqual(len(calibration.rounds), len(result.node.history))
        self.assertTrue(
            all(round_.modes_evaluated <= 4 for round_ in calibration.rounds)
        )
        self.assertTrue(
            all(round_.ranking_regret_cc >= 0 for round_ in calibration.rounds)
        )

    def test_history_calibration_uses_the_frozen_service_order_rule(self) -> None:
        distribution = (16, 15, 6, 5, 4, 3, 2, 2)
        result = paired_lpt_mode_search(
            distribution,
            beam_width=1,
            mode_budget=4,
            score_mode="local",
            service_order_mode="binding_chain",
        )
        calibration = calibrate_history_mode_choices(
            distribution,
            result.node,
            mode_budget=4,
            service_order_mode="binding_chain",
        )
        self.assertEqual(len(calibration.rounds), len(result.node.history))

    def test_history_calibration_uses_the_deployed_symmetric_bank(self) -> None:
        distribution = (16, 15, 6, 5, 4, 3, 2, 2)
        result = paired_lpt_mode_search(
            distribution,
            beam_width=1,
            score_mode="local",
            service_order_mode="binding_chain",
            tie_break_mode="bank_order",
            pair_mode_policy="no_mixed",
            mode_bank_policy="rtl_symmetric2",
        )
        calibration = calibrate_history_mode_choices(
            distribution,
            result.node,
            mode_budget=4,
            service_order_mode="binding_chain",
            pair_mode_policy="no_mixed",
            mode_bank_policy="rtl_symmetric2",
        )
        self.assertTrue(
            all(round_.modes_evaluated <= 2 for round_ in calibration.rounds)
        )


if __name__ == "__main__":
    unittest.main()
