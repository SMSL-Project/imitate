from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0
ITEMS = ()
POSITION_CODES = ()
ITEM_NAMES = {}
POSITION_NAMES = {}
REPO_OPERATION_NAME_OVERRIDES = {}
# Fill all constants with actual scenario data.
# REPO_OPERATION_NAME_OVERRIDES is for rare naming quirks only, not the primary mapping path.
# POSITION_NAMES should map symbolic destination codes to the exact repo naming tokens.
# Example: chess uses `A -> uppercase-A`, `a -> lowercase-a`, and `0 -> 0`.


def parse_state(state_name):
    # Parse one phase-state string into a structured state code.
    # Return the facts needed to apply an operation and validate the resulting transition.
    # Keep the structure compatible with encode_state(...).
    # Be careful with compact state names such as `State_ATAB`.
    # For fixed-width codes, remove the prefix and slice the payload instead of splitting on `_`.
    raise NotImplementedError


def encode_state(state_code):
    # Convert one structured state code back into the symbolic `State_...` name
    # used by the final SMSL JSON for this scenario.
    raise NotImplementedError


def parse_operation(operation_name):
    # Parse one symbolic phase operation into a structured operation code.
    # This parser must exactly match the symbolic `Op_...` format from phases 1-3.
    # Return the fields needed by both application and validation.
    # Match the parser to the actual operation syntax for this scenario.
    # If the format is delimited, such as `Op_star_B`, prefer `split("_", 2)`.
    # If the format is compact, such as `Op_1a`, parse the suffix after `Op_` by position.
    raise NotImplementedError


def apply_operation_under_state(state_code, operation_code):
    # Apply the symbolic operation to the structured state and return the resulting
    # structured next state.
    # Recompute the next state from scratch from the source state and operation.
    # Do not trust the carried target state from phase 3 here.
    # IMPORTANT: Update ONLY the fact(s) for the entity named by the operation.
    # Do not automatically move other entities (e.g., do not auto-move steve
    # when moving sheep, do not auto-move circle when moving star).
    # Each operation changes exactly one entity's position.
    raise NotImplementedError


def target_state_matches(current_target_state, computed_target_state):
    # Compare the carried target from phase 3 with the recomputed target from phase 4.
    # Most scenarios can simply use equality here.
    raise NotImplementedError


def transition_satisfies_rules(state_code, operation_code, next_state_code):
    # Return True only if the fully applied transition still obeys the puzzle rules.
    # This is the place to validate the resulting transition, not phase 2 or phase 3.
    # You MUST check ALL of the following:
    # 1. Source != target: the entity actually moved.
    # 2. Only the expected fact(s) changed: compare source and next state.
    # 3. The entity was at the operation's source location.
    # 4. The entity is now at the operation's target location.
    # 5. All puzzle-specific rules are satisfied in the resulting state.
    # 6. Any action preconditions (top-item, adjacency, etc.) are met.
    # If this returns False, phase 4 must remove the transition.
    raise NotImplementedError


def repo_operation_name(state_name, operation_name):
    # Map the symbolic phase operation into the final repo-native task name.
    # Build the generic mapping from the parsed operation whenever possible.
    # Use REPO_OPERATION_NAME_OVERRIDES only when the repo has naming quirks.
    # If overrides are state-dependent, key them by (state_name, operation_name).
    # Be precise about repo naming vocabulary.
    # For example, chess destinations are not all `uppercase-*`:
    # lowercase squares must use `lowercase-*`, and digit squares stay as `0` or `1`.
    raise NotImplementedError


def main():
    # Read `phase_3.json`, apply each surviving operation at its source state,
    # compare the recomputed target with the carried target, repair mismatches if needed,
    # drop invalid transitions, and write the final `phase_4.json`.
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
