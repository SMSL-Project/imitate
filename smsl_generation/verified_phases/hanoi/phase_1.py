from __future__ import annotations

import argparse
import json
from itertools import product


NUM_FACTS = 3
NUM_STATES = 27
INITIAL_STATE = "State_aaa"
ITEMS = ("1", "2", "3")
POSITION_CODES = ("a", "b", "c")


def base_position_codes():
    return POSITION_CODES


def expand_state_codes(state_code):
    yield tuple(state_code)


def iter_state_codes():
    for state_code in product(base_position_codes(), repeat=len(ITEMS)):
        yield from expand_state_codes(state_code)


def encode_state(state_code):
    return "State_" + "".join(state_code)


def build_candidate_operations():
    return ["Op_{}{}".format(item, position_code) for item in ITEMS for position_code in base_position_codes()]


def parse_operation(operation_name):
    return operation_name[3], operation_name[4]


def target_state_for_operation(state_code, operation_name):
    item, target_peg = parse_operation(operation_name)
    item_index = ITEMS.index(item)
    next_state = list(state_code)
    next_state[item_index] = target_peg
    return encode_state(tuple(next_state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="phase_1.json")
    args = parser.parse_args()

    operations = build_candidate_operations()
    sb = {
        "HEADER": {
            "INITIAL": INITIAL_STATE,
            "NUM_FACTS": NUM_FACTS,
            "NUM_STATES": NUM_STATES,
            "SUB_SBS": [],
        }
    }

    for state_code in iter_state_codes():
        state_name = encode_state(state_code)
        sb[state_name] = {
            operation_name: target_state_for_operation(state_code, operation_name)
            for operation_name in operations
        }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": sb}, fh, indent=2)


if __name__ == "__main__":
    main()
