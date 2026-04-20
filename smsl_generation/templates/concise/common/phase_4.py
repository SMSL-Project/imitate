from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0
ITEMS = ()
POSITION_CODES = ()

REPO_OPERATION_NAME_OVERRIDES = {}


def parse_state(state_name):
    raise NotImplementedError

def encode_state(state_code):
    raise NotImplementedError

def parse_operation(operation_name):
    raise NotImplementedError

def apply_operation_under_state(state_code, operation_code):
    raise NotImplementedError

def target_state_matches(computed_state_name, carried_state_name):
    return computed_state_name == carried_state_name

def transition_satisfies_rules(state_code, operation_code, next_state_code):
    raise NotImplementedError

def repo_operation_name(state_name, operation_name):
    if (state_name, operation_name) in REPO_OPERATION_NAME_OVERRIDES:
        return REPO_OPERATION_NAME_OVERRIDES[(state_name, operation_name)]
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase_3.json")
    parser.add_argument("--output", default="phase_4.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sb = data["SB1"]
    final = {"HEADER": dict(sb["HEADER"])}
    for state_name, transitions in sb.items():
        if state_name == "HEADER":
            continue
        state_code = parse_state(state_name)
        final[state_name] = {}
        for operation_name, carried_target_state in transitions.items():
            operation_code = parse_operation(operation_name)
            computed_next = apply_operation_under_state(state_code, operation_code)
            computed_next_name = encode_state(computed_next)
            if not target_state_matches(computed_next_name, carried_target_state):
                continue
            next_state_code = parse_state(computed_next_name)
            if not transition_satisfies_rules(state_code, operation_code, next_state_code):
                continue
            final[state_name][repo_operation_name(state_name, operation_name)] = carried_target_state

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": final}, fh, indent=2)


if __name__ == "__main__":
    main()
