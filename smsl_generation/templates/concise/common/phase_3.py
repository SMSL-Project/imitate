from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0
ITEMS = ()
POSITION_CODES = ()


def parse_state(state_name):
    raise NotImplementedError

def parse_operation(operation_name):
    raise NotImplementedError

def operation_is_possible_under_state(state_code, operation_code):
    raise NotImplementedError


def main():
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
