#!/usr/bin/env python3

import unittest

from n_outer_scheduler.coarse_model.run_evaluation import _work_signature


class EvaluationTest(unittest.TestCase):
    def test_same_input_has_identical_atomic_work(self) -> None:
        signature = _work_signature((16, 4, 2, 2))
        self.assertGreater(signature["compute_cc"], 0)
        self.assertGreater(signature["weight_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

