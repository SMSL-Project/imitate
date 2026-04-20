from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0
ITEMS = ()
POSITION_CODES = ()
# Fill these with the actual scenario constants used by the parser and legality checks.


def parse_state(state_name):
    # Parse one phase-state string into a structured state code.
    # Return the facts needed to decide whether an operation is executable here.
    # Use the same item ordering and code types consistently across the file.
    # Be careful with compact state names such as `State_ATAB`.
    # For fixed-width codes, remove the prefix and slice the payload instead of splitting on `_`.
    raise NotImplementedError


def parse_operation(operation_name):
    # Parse one symbolic phase operation into a structured operation code.
    # This parser must exactly match the symbolic `Op_...` format from phase 1.
    # Return the fields needed by `operation_is_possible_under_state(...)`.
    # Match the parser to the actual operation syntax for this scenario.
    # If the format is delimited, such as `Op_star_B`, prefer `split("_", 2)`.
    # If the format is compact, such as `Op_1a`, parse the suffix after `Op_` by position.
    raise NotImplementedError


def operation_is_possible_under_state(state_code, operation_code):
    # Return True only if this operation can actually be executed from this state.
    # This function must check ALL of the following:
    # 1. Source != target: the operation actually changes something.
    # 2. The entity is at the source and can be moved (e.g., top disk, steve proximity).
    # 3. Adjacency / connectivity: the source and target are directly connected.
    # 4. Any prerequisite constraints from the puzzle rules.
    # Keep the carried target state untouched here.
    # Do not rename states or map to repo-native task names in this phase.
    # If the puzzle uses a board or graph, derive legal neighbors carefully.
    # "Adjacent valid positions" means directly connected valid cells only;
    # blocked cells do not create pass-through adjacency.
    raise NotImplementedError


def main():
    # Read `phase_2.json`, remove operations that are impossible under their states,
    # and write `phase_3.json` with the surviving carried target states unchanged.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase_2.json")
    parser.add_argument("--output", default="phase_3.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sb = data["SB1"]
    filtered = {"HEADER": dict(sb["HEADER"])}
    for state_name, transitions in sb.items():
        if state_name == "HEADER":
            continue
        state_code = parse_state(state_name)
        filtered[state_name] = {}
        for operation_name, next_state in transitions.items():
            operation_code = parse_operation(operation_name)
            if operation_is_possible_under_state(state_code, operation_code):
                filtered[state_name][operation_name] = next_state

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": filtered}, fh, indent=2)


if __name__ == "__main__":
    main()
