from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0


def parse_state(state_name):
    raise NotImplementedError

def state_satisfies_rules(state_code):
    raise NotImplementedError

def canonical_state_name(state_name):
    raise NotImplementedError

def is_valid_state(state_name, transitions):
    del transitions
    state_code = parse_state(state_name)
    return state_satisfies_rules(state_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase_1.json")
    parser.add_argument("--output", default="phase_2.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sb = data["SB1"]
    filtered = {"HEADER": dict(sb["HEADER"])}
    for state_name, transitions in sb.items():
        if state_name == "HEADER":
            continue
        if is_valid_state(state_name, transitions):
            filtered[canonical_state_name(state_name)] = transitions

    filtered["HEADER"]["NUM_FACTS"] = NUM_FACTS
    filtered["HEADER"]["NUM_STATES"] = NUM_STATES

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": filtered}, fh, indent=2)


if __name__ == "__main__":
    main()
