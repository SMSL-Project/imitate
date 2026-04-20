from __future__ import annotations

import argparse
import json
from itertools import product


NUM_FACTS = 2
NUM_STATES = 324
INITIAL_STATE = "State_0B1B"
ITEMS = ("star", "circle")
POSITION_CODES = ("AT", "AB", "BT", "BB", "CT", "CB", "DT", "DB", "aT", "aB", "cT", "cB", "dT", "dB", "0T", "0B", "1T", "1B")


def base_position_codes():
    return tuple(
        dict.fromkeys(
            position_code[:-1]
            for position_code in POSITION_CODES
        )
    )


def expand_state_codes(state_code):
    yield tuple(state_code)


def iter_state_codes():
    for state_code in product(POSITION_CODES, repeat=len(ITEMS)):
        yield from expand_state_codes(state_code)


def encode_state(state_code):
    star_code, circle_code = state_code
    return "State_{}{}".format(star_code, circle_code)


def build_candidate_operations():
    return ["Op_{}_{}".format(item, position_code) for item in ITEMS for position_code in base_position_codes()]


def parse_operation(operation_name):
    _, piece, destination = operation_name.split("_", 2)
    return piece, destination


def target_state_for_operation(state_code, operation_name):
    piece, destination = parse_operation(operation_name)
    item_index = ITEMS.index(piece)
    next_state = list(state_code)
    next_state[item_index] = destination + state_code[item_index][-1]
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
