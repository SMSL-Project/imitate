from __future__ import annotations

import argparse
import json
from itertools import product


NUM_FACTS = 0
NUM_STATES = 0
INITIAL_STATE = "State_REPLACE_ME"
ITEMS = ()
POSITION_CODES = ()


def base_position_codes():
    raise NotImplementedError

def expand_state_codes(state_code):
    raise NotImplementedError

def iter_state_codes():
    for state_code in product(base_position_codes(), repeat=len(ITEMS)):
        yield from expand_state_codes(state_code)

def encode_state(state_code):
    raise NotImplementedError

def build_candidate_operations():
    raise NotImplementedError

def parse_operation(operation_name):
    raise NotImplementedError

def target_state_for_operation(state_code, operation_name):
    raise NotImplementedError


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
