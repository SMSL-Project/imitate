from __future__ import annotations

import argparse
import json


NUM_FACTS = 2
NUM_STATES = 90
ITEMS = ("star", "circle")
POSITION_CODES = ("AT", "AB", "BT", "BB", "CT", "CB", "DT", "DB", "aT", "aB", "cT", "cB", "dT", "dB", "0T", "0B", "1T", "1B")
ITEM_NAMES = {item: item for item in ITEMS}
POSITION_NAMES = {
    "0": "0",
    "1": "1",
    "a": "lowercase-a",
    "c": "lowercase-c",
    "d": "lowercase-d",
    "A": "uppercase-A",
    "B": "uppercase-B",
    "C": "uppercase-C",
    "D": "uppercase-D",
}
REPO_OPERATION_NAME_OVERRIDES = {}


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {
        "star": payload[:2],
        "circle": payload[2:],
    }


def encode_state(state_code):
    return "State_" + "".join(state_code[item] for item in ITEMS)


def parse_operation(operation_name):
    _, item, target_code = operation_name.split("_", 2)
    return {
        "item": item,
        "source": None,
        "target": target_code,
    }


def apply_operation_under_state(state_code, operation_code):
    item = operation_code["item"]
    target_position = operation_code["target"]
    star_position = state_code["star"][0]
    circle_position = state_code["circle"][0]

    if item == "star":
        new_star_position = target_position
        new_circle_position = circle_position
    else:
        new_star_position = star_position
        new_circle_position = target_position

    if new_star_position != new_circle_position:
        return {"star": new_star_position + "B", "circle": new_circle_position + "B"}
    if item == "star":
        return {"star": new_star_position + "T", "circle": new_circle_position + "B"}
    return {"star": new_star_position + "B", "circle": new_circle_position + "T"}


def target_state_matches(current_target_state, computed_target_state):
    return current_target_state == computed_target_state


def transition_satisfies_rules(state_code, operation_code, next_state_code):
    item = operation_code["item"]
    target_position = operation_code["target"]
    star_code = state_code["star"]
    circle_code = state_code["circle"]
    next_star_code = next_state_code["star"]
    next_circle_code = next_state_code["circle"]
    star_position = star_code[0]
    circle_position = circle_code[0]
    next_star_position = next_star_code[0]
    next_circle_position = next_circle_code[0]

    if star_code.endswith("T"):
        top_item = "star"
    elif circle_code.endswith("T"):
        top_item = "circle"
    else:
        top_item = None

    if next_star_code.endswith("T"):
        next_top_item = "star"
    elif next_circle_code.endswith("T"):
        next_top_item = "circle"
    else:
        next_top_item = None

    current_position = star_position if item == "star" else circle_position

    if top_item is not None and item != top_item:
        return False
    if item == "star":
        if next_circle_position != circle_position:
            return False
        if next_star_position != target_position:
            return False
    else:
        if next_star_position != star_position:
            return False
        if next_circle_position != target_position:
            return False
    if next_star_position == next_circle_position:
        if next_top_item != item:
            return False
    else:
        if not next_star_code.endswith("B") or not next_circle_code.endswith("B"):
            return False
        if next_top_item is not None:
            return False
    if target_position == current_position:
        return False
    return True


def repo_operation_name(state_name, operation_name):
    if (state_name, operation_name) in REPO_OPERATION_NAME_OVERRIDES:
        return REPO_OPERATION_NAME_OVERRIDES[(state_name, operation_name)]

    operation_code = parse_operation(operation_name)
    return "move-{}-chess-to-block-{}".format(
        ITEM_NAMES[operation_code["item"]],
        POSITION_NAMES[operation_code["target"]],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase_3.json")
    parser.add_argument("--output", default="phase_4.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sb = data["SB1"]
    valid_states = {state_name for state_name in sb if state_name != "HEADER"}
    completed = {"HEADER": dict(sb["HEADER"])}
    for state_name, transitions in sb.items():
        if state_name == "HEADER":
            continue
        completed[state_name] = {}
        for operation_name, current_target_state in transitions.items():
            state_code = parse_state(state_name)
            operation_code = parse_operation(operation_name)
            next_state_code = apply_operation_under_state(state_code, operation_code)
            computed_target_state = encode_state(next_state_code)
            if target_state_matches(current_target_state, computed_target_state):
                resolved_target_state = current_target_state
            else:
                resolved_target_state = computed_target_state
            if resolved_target_state not in valid_states:
                continue
            if transition_satisfies_rules(state_code, operation_code, next_state_code):
                completed[state_name][repo_operation_name(state_name, operation_name)] = resolved_target_state

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"SB1": completed}, fh, indent=2)


if __name__ == "__main__":
    main()
