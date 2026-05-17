import unittest
from collections import Counter

from smsl_gensim import available_scenarios, check_demo_plan, iter_transitions


class DemoPlanTests(unittest.TestCase):
    def test_transition_counts(self):
        expected = {
            "chess": 324,
            "hanoi": 78,
            "river_crossing": 92,
        }
        for scenario, count in expected.items():
            self.assertEqual(len(tuple(iter_transitions(scenario))), count)

    def test_all_demo_plans_are_complete(self):
        for scenario in available_scenarios():
            check = check_demo_plan(scenario)
            self.assertTrue(check.ok, check)

    def test_hanoi_collects_transitions_not_only_task_names(self):
        transitions = tuple(iter_transitions("hanoi"))
        task_counts = Counter(t.task_name for t in transitions)
        self.assertEqual(len(task_counts), 9)
        self.assertEqual(len(transitions), 78)
        self.assertGreater(task_counts["move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand"], 1)


if __name__ == "__main__":
    unittest.main()
