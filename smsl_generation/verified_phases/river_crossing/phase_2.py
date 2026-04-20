from __future__ import annotations

import argparse
import json


NUM_FACTS = 4
NUM_STATES = 40


def parse_state(state_name):
    return tuple(state_name.replace("State_", "", 1))


def state_satisfies_rules(state_code):
    grass, sheep, steve, wolf = state_code
    boat_count = state_code.count("B")

    if sheep == wolf != steve and sheep != "B":
        return False
    if sheep == grass != steve and sheep != "B":
        return False
    if boat_count > 2:
        return False
    if boat_count == 2 and steve != "B":
        return False
    return True


def canonical_state_name(state_name):
    state_code = parse_state(state_name)
    if not state_satisfies_rules(state_code):
        return None
    return "State_" + "".join(state_code)


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
