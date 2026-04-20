from __future__ import annotations

import argparse
import json


NUM_FACTS = 4
NUM_STATES = 40
ITEMS = ("grass", "sheep", "steve", "wolf")
POSITION_CODES = ("0", "B", "1")


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {item: position_code for item, position_code in zip(ITEMS, payload)}


def parse_operation(operation_name):
    _, item, source_code, target_code = operation_name.split("_")
    return {
        "item": item,
        "source": source_code,
        "target": target_code,
    }


def operation_is_possible_under_state(state_code, operation_code):
    adjacent_moves = {("0", "B"), ("B", "0"), ("B", "1"), ("1", "B")}
    item = operation_code["item"]
    source_code = operation_code["source"]
    target_code = operation_code["target"]
    steve_code = state_code["steve"]

    if (source_code, target_code) not in adjacent_moves:
        return False
    if state_code[item] != source_code:
        return False
    if item == "steve":
        return steve_code == source_code
    if steve_code == "B":
        return True
    if steve_code == "0":
        return "1" not in {source_code, target_code}
    if steve_code == "1":
        return "0" not in {source_code, target_code}
    return False


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
