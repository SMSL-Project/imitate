import unittest
from pathlib import Path

from smsl_gensim import (
    CollectionJob,
    CollectionResult,
    build_collection_plan,
    collect_demos,
)
from smsl_gensim.__main__ import _collection_result_line


class CollectionTests(unittest.TestCase):
    def test_collection_plan_is_per_transition(self):
        plan = build_collection_plan("hanoi", demos_per_transition=2, split="val")
        job = plan.jobs[0]
        self.assertEqual(plan.transition_count, 78)
        self.assertEqual(plan.total_demos, 156)
        self.assertEqual(job.from_state, "State_aaa")
        self.assertEqual(job.operation, "move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand")
        self.assertEqual(job.to_state, "State_baa")
        self.assertEqual(job.task_name, "move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand")
        self.assertEqual(job.split, "val")
        self.assertEqual(job.demo_index, 0)
        self.assertEqual(
            job.output_dir,
            Path("data/smsl/hanoi/val/State_aaa__move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand__State_baa"),
        )

    def test_output_dirs_are_filesystem_safe(self):
        plan = build_collection_plan("chess")
        for job in plan.jobs:
            self.assertNotIn("/", job.output_dir.name)
            self.assertNotIn(" ", job.output_dir.name)

    def test_collect_demos_counts_unsaved_results(self):
        plan = build_collection_plan("hanoi", demos_per_transition=1)
        calls = []

        def fake_collector(job, assets_root="", disp=False):
            calls.append(job)
            saved = len(calls) != 1
            return CollectionResult(job, seed=job.demo_index + 2, total_reward=float(saved), saved=saved)

        run = collect_demos(plan, collector=fake_collector)
        self.assertEqual(len(run.results), 78)
        self.assertEqual(run.saved_count, 77)
        self.assertEqual(run.unsaved_count, 1)

    def test_collection_result_line_includes_failure_reason(self):
        job = CollectionJob(
            scenario="hanoi",
            split="train",
            demo_index=0,
            from_state="State_aaa",
            operation="move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand",
            to_state="State_baa",
            task_name="move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand",
            output_dir=Path("data/smsl/hanoi/train/example"),
        )
        result = CollectionResult(job, seed=2, total_reward=0.0, saved=False, failure_reason="oracle missed object")

        line = _collection_result_line(1, 1, result)

        self.assertIn("saved=False", line)
        self.assertIn("reason=oracle missed object", line)


if __name__ == "__main__":
    unittest.main()
