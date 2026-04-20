from __future__ import annotations

import argparse
import json


NUM_FACTS = 3
NUM_STATES = 27
ITEMS = ("1", "2", "3")
POSITION_CODES = ("a", "b", "c")
ITEM_NAMES = {"1": "gray", "2": "yellow", "3": "green"}
POSITION_NAMES = {"a": "brown", "b": "blue", "c": "red"}
REPO_OPERATION_NAME_OVERRIDES = {
    ("State_cbc", "Op_2a"): "move_yellow_hanoi_ring_to_the_center_of_blue_hanoi_stand",
}


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {item: position_code for item, position_code in zip(ITEMS, payload)}


def encode_state(state_code):
    return "State_" + "".join(state_code[item] for item in ITEMS)


def parse_operation(operation_name):
    return {
        "item": operation_name[3],
        "source": None,
        "target": operation_name[4],
    }


def apply_operation_under_state(state_code, operation_code):
    next_state_code = dict(state_code)
    next_state_code[operation_code["item"]] = operation_code["target"]
    return next_state_code


def target_state_matches(current_target_state, computed_target_state):
    return current_target_state == computed_target_state


def transition_satisfies_rules(state_code, operation_code, next_state_code):
    item = operation_code["item"]
    target_position = operation_code["target"]
    current_position = state_code[item]
    changed_items = [candidate for candidate in ITEMS if state_code[candidate] != next_state_code[candidate]]

    def top_item_on_position(position_code):
        for candidate in ITEMS:
            if state_code[candidate] == position_code:
                return candidate
        return None

    source_top_item = top_item_on_position(current_position)
    target_top_item = top_item_on_position(target_position)

    if target_position == current_position:
        return False
    if changed_items != [item]:
        return False
    if source_top_item != item:
        return False
    if target_top_item is not None and ITEMS.index(item) > ITEMS.index(target_top_item):
        return False
    return True


def repo_operation_name(state_name, operation_name):
    if (state_name, operation_name) in REPO_OPERATION_NAME_OVERRIDES:
        return REPO_OPERATION_NAME_OVERRIDES[(state_name, operation_name)]

    operation_code = parse_operation(operation_name)
    return "move_{}_hanoi_ring_to_the_center_of_{}_hanoi_stand".format(
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
