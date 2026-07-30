#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

from n_outer_scheduler.coarse_model.analyze_final_policy_eval import summarize


class FinalPolicyAnalysisTests(unittest.TestCase):
    def test_one_case_smoke_output_is_analyzable(self):
        path = Path("/tmp/n_outer_final_eval_smoke.json")
        if not path.exists():
            self.skipTest("standalone smoke artifact is absent")
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = summarize(payload)
        self.assertEqual(summary["case_count"], 1)
        self.assertFalse(summary["complete_65"])
        self.assertTrue(summary["all_histories_and_task_replays_valid"])


if __name__ == "__main__":
    unittest.main()
