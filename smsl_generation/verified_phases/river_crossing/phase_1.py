from __future__ import annotations

import argparse
import json
from itertools import product


NUM_FACTS = 4
NUM_STATES = 81
INITIAL_STATE = "State_0000"
ITEMS = ("grass", "sheep", "steve", "wolf")
POSITION_CODES = ("0", "B", "1")


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
    adjacent_moves = (("0", "B"), ("B", "0"), ("B", "1"), ("1", "B"))
    return [
        "Op_{}_{}_{}".format(item, source_code, target_code)
        for item in ITEMS
        for source_code, target_code in adjacent_moves
    ]


def parse_operation(operation_name):
    _, item, source_code, target_code = operation_name.split("_")
    return item, source_code, target_code


def target_state_for_operation(state_code, operation_name):
    item, source_code, target_code = parse_operation(operation_name)
    del source_code
    next_state = list(state_code)
    item_index = ITEMS.index(item)
    next_state[item_index] = target_code
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
