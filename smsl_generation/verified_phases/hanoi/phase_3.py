from __future__ import annotations

import argparse
import json


NUM_FACTS = 3
NUM_STATES = 27
ITEMS = ("1", "2", "3")
POSITION_CODES = ("a", "b", "c")


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {item: position_code for item, position_code in zip(ITEMS, payload)}


def parse_operation(operation_name):
    return {
        "item": operation_name[3],
        "source": None,
        "target": operation_name[4],
    }


def operation_is_possible_under_state(state_code, operation_code):
    item = operation_code["item"]
    target_position = operation_code["target"]
    current_position = state_code[item]

    def top_item_on_position(position_code):
        for candidate in ITEMS:
            if state_code[candidate] == position_code:
                return candidate
        return None

    source_top_item = top_item_on_position(current_position)
    target_top_item = top_item_on_position(target_position)

    if target_position == current_position:
        return False
    if source_top_item != item:
        return False
    if target_top_item is not None and ITEMS.index(item) > ITEMS.index(target_top_item):
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
