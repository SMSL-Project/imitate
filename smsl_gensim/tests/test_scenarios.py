import unittest

from smsl_gensim import available_scenarios, load_smsl, load_task_list, operation_names


class ScenarioTests(unittest.TestCase):
    def test_task_lists_match_smsl_operations(self):
        for name in available_scenarios():
            tasks = set(load_task_list(name))
            operations = {op.replace("_", "-") for op in operation_names(name)}
            self.assertEqual(tasks, operations)

    def test_smsl_headers_are_consistent(self):
        for name in available_scenarios():
            graph = next(iter(load_smsl(name).values()))
            states = [state for state in graph if state != "HEADER"]
            self.assertEqual(graph["HEADER"]["NUM_STATES"], len(states))

    def test_hanoi_operation_names_match_state_changes(self):
        graph = next(iter(load_smsl("hanoi").values()))
        for from_state, edges in graph.items():
            if from_state == "HEADER":
                continue
            for operation, to_state in edges.items():
                disk, stand = _hanoi_operation(operation)
                changed = _hanoi_changed_disk(from_state, to_state)
                self.assertEqual(disk, changed)
                self.assertEqual(stand, HANOI_STANDS[_hanoi_payload(to_state)[HANOI_DISKS.index(disk)]])

HANOI_DISKS = ("gray", "yellow", "green")
HANOI_STANDS = {"a": "brown", "b": "blue", "c": "red"}


def _hanoi_payload(state):
    return state.replace("State_", "", 1)


def _hanoi_operation(operation):
    parts = operation.split("_")
    return parts[1], parts[8]


def _hanoi_changed_disk(from_state, to_state):
    changes = [
        disk
        for disk, before, after in zip(HANOI_DISKS, _hanoi_payload(from_state), _hanoi_payload(to_state))
        if before != after
    ]
    return changes[0]


if __name__ == "__main__":
    unittest.main()
