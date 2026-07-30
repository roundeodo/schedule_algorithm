#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.candidate_adapter import inject_candidate
from n_outer_scheduler.coarse_model.candidates import materialize_modes
from n_outer_scheduler.coarse_model.semantics import ActionKind


class CandidateAdapterTest(unittest.TestCase):
    def test_external_pair_imports_only_slice_semantics(self) -> None:
        skeleton = inject_candidate(
            "PAIR", ((0, 1, 0, 8), (1, 5, 0, 2))
        )
        self.assertEqual(skeleton.kind, ActionKind.PAIR)
        modes = materialize_modes(skeleton)
        self.assertTrue(modes)
        self.assertTrue(all(mode.tasks[0].expert_slice.eid == 1 for mode in modes))


if __name__ == "__main__":
    unittest.main()

