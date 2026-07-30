#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.block_golden import validate_block_result
from n_outer_scheduler.coarse_model.candidates import WindowSpec
from n_outer_scheduler.coarse_model.lowering import (
    expand_bingo_program,
    lower_history_to_bingo,
    replay_bingo_program,
)
from n_outer_scheduler.coarse_model.search import SearchConfig, beam_search


class LoweringTest(unittest.TestCase):
    def test_lowered_program_replays_without_source_history(self) -> None:
        distribution = (8, 2)
        result = beam_search(
            distribution,
            config=SearchConfig(
                window=WindowSpec(2, 0),
                beam_width=8,
                candidate_budget=16,
            ),
        )
        program = lower_history_to_bingo(distribution, result.node)
        streams = expand_bingo_program(program)
        self.assertEqual(
            len(streams[0]) + len(streams[1]), 16 * len(program.records)
        )
        replay = replay_bingo_program(program)
        validate_block_result(replay)
        self.assertTrue(replay.history_validated)

    def test_split_token_ranges_remain_exact_in_bingo_records(self) -> None:
        distribution = (3,)
        result = beam_search(
            distribution,
            config=SearchConfig(
                window=WindowSpec(1, 0),
                beam_width=8,
                candidate_budget=16,
            ),
        )
        program = lower_history_to_bingo(distribution, result.node)
        ranges = sorted(
            (record.token_start, record.token_end) for record in program.records
        )
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 3)
        self.assertEqual(sum(end - start for start, end in ranges), 3)


if __name__ == "__main__":
    unittest.main()
