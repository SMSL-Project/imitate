from __future__ import annotations

import argparse
import json


NUM_FACTS = 2
NUM_STATES = 90
ITEMS = ("star", "circle")
POSITION_CODES = ("AT", "AB", "BT", "BB", "CT", "CB", "DT", "DB", "aT", "aB", "cT", "cB", "dT", "dB", "0T", "0B", "1T", "1B")


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {
        "star": payload[:2],
        "circle": payload[2:],
    }


def parse_operation(operation_name):
    _, item, target_code = operation_name.split("_", 2)
    return {
        "item": item,
        "source": None,
        "target": target_code,
    }


def operation_is_possible_under_state(state_code, operation_code):
    valid_moves = {
        "0": ("a",),
        "1": ("d",),
        "A": ("B", "a"),
        "B": ("A", "C"),
        "C": ("B", "D", "c"),
        "D": ("C", "d"),
        "a": ("0", "A"),
        "c": ("C", "d"),
        "d": ("1", "c", "D"),
    }
    item = operation_code["item"]
    target_code = operation_code["target"]
    star_code = state_code["star"]
    circle_code = state_code["circle"]

    if star_code.endswith("T"):
        top_item = "star"
    elif circle_code.endswith("T"):
        top_item = "circle"
    else:
        top_item = None

    current_position = state_code[item][0]
    if top_item is not None and item != top_item:
        return False
    if target_code not in valid_moves[current_position]:
        return False
    return True


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
        filtered[state_name] = {}
        for operation_name, next_state in transitions.items():
            state_code = parse_state(state_name)
            operation_code = parse_operation(operation_name)
            if operation_is_possible_under_state(state_code, operation_code):
                filtered[state_name][operation_name] = next_state

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": filtered}, fh, indent=2)


if __name__ == "__main__":
    main()
