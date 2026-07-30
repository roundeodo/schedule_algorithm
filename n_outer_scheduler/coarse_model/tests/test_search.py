#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.candidates import WindowSpec
from n_outer_scheduler.coarse_model.search import (
    SearchConfig,
    beam_search,
    continuation_lower_bound,
    continuation_lpt_estimate,
    remaining_from_distribution,
    validate_history,
)
from n_outer_scheduler.coarse_model.semantics import MacroScheduleState


class SearchTest(unittest.TestCase):
    def test_continuation_bound_is_not_below_committed_history(self) -> None:
        state = MacroScheduleState(
            cluster_free_cc=(100, 80),
            prefetch_release_cc=(90, 70),
        )
        bound = continuation_lower_bound(
            state, remaining_from_distribution((4, 2))
        )
        self.assertGreaterEqual(bound, 100)

    def test_lpt_estimate_retains_indivisible_hot_experts(self) -> None:
        state = MacroScheduleState()
        remaining = remaining_from_distribution((16, 15, 2, 2))
        self.assertGreaterEqual(
            continuation_lpt_estimate(state, remaining),
            continuation_lower_bound(state, remaining),
        )

    def test_small_search_completes_and_replays_exactly(self) -> None:
        distribution = (4, 2)
        result = beam_search(
            distribution,
            config=SearchConfig(
                window=WindowSpec(2, 0),
                beam_width=8,
                candidate_budget=16,
            ),
        )
        self.assertTrue(result.history_validated)
        self.assertFalse(result.node.remaining)
        validate_history(distribution, result.node)
        self.assertGreater(result.node.makespan_cc, 0)


if __name__ == "__main__":
    unittest.main()
