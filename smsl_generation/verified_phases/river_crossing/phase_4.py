from __future__ import annotations

import argparse
import json


NUM_FACTS = 4
NUM_STATES = 40
ITEMS = ("grass", "sheep", "steve", "wolf")
POSITION_CODES = ("0", "B", "1")
ITEM_NAMES = {item: item for item in ITEMS}
POSITION_NAMES = {"0": "red_land", "B": "boat", "1": "green_land"}
REPO_OPERATION_NAME_OVERRIDES = {}


def parse_state(state_name):
    payload = state_name.replace("State_", "", 1)
    return {item: position_code for item, position_code in zip(ITEMS, payload)}


def encode_state(state_code):
    return "State_" + "".join(state_code[item] for item in ITEMS)


def parse_operation(operation_name):
    _, item, source_code, target_code = operation_name.split("_")
    return {
        "item": item,
        "source": source_code,
        "target": target_code,
    }


def apply_operation_under_state(state_code, operation_code):
    next_state_code = dict(state_code)
    next_state_code[operation_code["item"]] = operation_code["target"]
    return next_state_code


def target_state_matches(current_target_state, computed_target_state):
    return current_target_state == computed_target_state


def transition_satisfies_rules(state_code, operation_code, next_state_code):
    item = operation_code["item"]
    source_code = operation_code["source"]
    target_code = operation_code["target"]
    changed_items = [candidate for candidate in ITEMS if state_code[candidate] != next_state_code[candidate]]
    boat_count = tuple(next_state_code.values()).count("B")

    if changed_items != [item]:
        return False
    if state_code[item] != source_code:
        return False
    if next_state_code[item] != target_code:
        return False
    if boat_count > 2:
        return False
    if boat_count == 2 and next_state_code["steve"] != "B":
        return False
    if next_state_code["sheep"] == next_state_code["wolf"] != next_state_code["steve"] and next_state_code["sheep"] != "B":
        return False
    if next_state_code["sheep"] == next_state_code["grass"] != next_state_code["steve"] and next_state_code["sheep"] != "B":
        return False
    return True


def repo_operation_name(state_name, operation_name):
    if (state_name, operation_name) in REPO_OPERATION_NAME_OVERRIDES:
        return REPO_OPERATION_NAME_OVERRIDES[(state_name, operation_name)]

    operation_code = parse_operation(operation_name)
    return "move_{}_from_{}_to_{}".format(
        ITEM_NAMES[operation_code["item"]],
        POSITION_NAMES[operation_code["source"]],
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
