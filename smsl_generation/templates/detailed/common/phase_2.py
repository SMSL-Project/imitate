from __future__ import annotations

import argparse
import json


NUM_FACTS = 0
NUM_STATES = 0
# Fill these with the real counts after phase 2 finishes.
# Do not leave placeholders such as 0 in the final script.


def parse_state(state_name):
    # Parse one phase-state string into a structured state code for rule checking.
    # The returned value should contain only the facts needed by phase 2 state validation.
    # Keep the representation simple and consistent with the phase 1 encoding.
    # Important: parse only what the state name actually encodes.
    # Do not invent extra hidden facts such as move history, source peg, or stack order
    # unless those facts are explicitly present in the state syntax.
    # Be careful with compact state names such as `State_ATAB`.
    # That example has only the `State_` underscore, so do not expect `State_AT_AB`.
    # For fixed-width piece codes, remove the `State_` prefix and slice the payload.
    raise NotImplementedError


def state_satisfies_rules(state_code):
    # Return True only if this structured state can exist as a snapshot of the puzzle.
    # Phase 2 filters states only.
    # Do not check whether a move into or out of this state is legal.
    # Do not use operation preconditions such as adjacency, top-item checks, or source!=target here.
    # Example: if the encoding already implies a legal stack order, then all assignments may be valid.
    # Another way to think about it:
    # ask "can this arrangement of encoded facts exist right now?"
    # not "could a particular move have produced it?"
    # If the encoding only says where each item is, then all item-location assignments
    # may be valid even when some moves between them are illegal.
    # Example for Hanoi:
    # `State_abc`, `State_cba`, and every other 3-letter peg assignment are valid states.
    # The restrictions "only move the top disk" and "do not place a larger disk on a smaller one"
    # belong to phases 3 and 4, not here.
    # Example for river crossing:
    # phase 2 must enforce both safety rules and occupancy rules such as boat capacity.
    # If the state syntax explicitly includes who is on the boat, then
    # constraints like "boat can carry at most two entities" and
    # "if two entities are on the boat, one must be Steve" belong here.
    raise NotImplementedError


def canonical_state_name(state_name):
    # Return the state name that should be kept in phase 2.
    # Usually this rebuilds the name from `parse_state(...)`.
    # Return the original state name when no renaming is needed.
    # Return None when the state is invalid and should be removed.
    # Do not silently drop information here.
    # If no normalization is needed, return the original state name unchanged.
    raise NotImplementedError


def is_valid_state(state_name, transitions):
    # Keep this wrapper so the phase runner has one state-level predicate.
    # `transitions` is available if the encoding needs it, but phase 2 should
    # normally validate from the state itself.
    # Avoid depending on transitions unless the state syntax truly requires them.
    del transitions
    state_code = parse_state(state_name)
    return state_satisfies_rules(state_code)


def main():
    # Read `phase_1.json`, remove invalid states, and write `phase_2.json`.
    # This phase should only shrink or normalize the state inventory.
    # After filtering, set HEADER.NUM_STATES to the true kept-state count for phase 2.
    # If phase 2 keeps every state, preserve the full count instead of inventing a smaller one.
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
