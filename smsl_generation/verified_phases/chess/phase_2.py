from __future__ import annotations

import argparse
import json


NUM_FACTS = 2
NUM_STATES = 90


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return payload[:2], payload[2:]


def state_satisfies_rules(state_code):
    star_code, circle_code = state_code
    star_position, circle_position = star_code[0], circle_code[0]
    star_on_top = star_code.endswith("T")
    circle_on_top = circle_code.endswith("T")
    star_on_bottom = star_code.endswith("B")
    circle_on_bottom = circle_code.endswith("B")

    if star_position != circle_position:
        return star_on_bottom and circle_on_bottom
    return (star_on_top and circle_on_bottom) or (circle_on_top and star_on_bottom)


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
