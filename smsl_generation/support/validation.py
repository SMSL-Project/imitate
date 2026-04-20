from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from smsl_generation.support.files import read_json
from smsl_generation.support.scenarios import get_scenario_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_DIR = REPO_ROOT / "smsl_gensim" / "smsl_gensim" / "SMSL_JSON"


@dataclass(frozen=True)
class ValidationDiagnostics:
    scenario_name: str
    expected_state_count: int
    actual_state_count: int
    missing_states: tuple[str, ...]
    extra_states: tuple[str, ...]


class SMSLValidationError(ValueError):
    def __init__(self, message: str, diagnostics: ValidationDiagnostics | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class SMSLValidator:
    @staticmethod
    def validate_shape(smsl: Dict[str, Any], scenario_name: str) -> None:
        if not isinstance(smsl, dict) or set(smsl.keys()) != {"SB1"}:
            raise SMSLValidationError("SMSL JSON must have exactly one top-level key: `SB1`.")

        body = smsl["SB1"]
        if not isinstance(body, dict) or "HEADER" not in body:
            raise SMSLValidationError("SMSL JSON must contain `SB1.HEADER`.")

        scenario = get_scenario_spec(scenario_name)
        header = body["HEADER"]
        required_header = {"INITIAL", "NUM_FACTS", "NUM_STATES", "SUB_SBS"}
        if not isinstance(header, dict) or set(header.keys()) != required_header:
            raise SMSLValidationError("`HEADER` must contain `INITIAL`, `NUM_FACTS`, `NUM_STATES`, and `SUB_SBS`.")

        if not isinstance(header["INITIAL"], str) or not scenario.state_name_is_valid(header["INITIAL"]):
            raise SMSLValidationError("`HEADER.INITIAL` must be a string.")
        if not isinstance(header["NUM_FACTS"], int):
            raise SMSLValidationError("`HEADER.NUM_FACTS` must be an integer.")
        if not isinstance(header["NUM_STATES"], int):
            raise SMSLValidationError("`HEADER.NUM_STATES` must be an integer.")
        if not isinstance(header["SUB_SBS"], list):
            raise SMSLValidationError("`HEADER.SUB_SBS` must be a list.")

        if header["NUM_FACTS"] != scenario.num_facts:
            raise SMSLValidationError(
                "`HEADER.NUM_FACTS` must be {} for scenario `{}`, but found {}.".format(
                    scenario.num_facts,
                    scenario.name,
                    header["NUM_FACTS"],
                )
            )
        if header["NUM_STATES"] != scenario.state_count:
            raise SMSLValidationError(
                "`HEADER.NUM_STATES` must be {} for scenario `{}`, but found {}.".format(
                    scenario.state_count,
                    scenario.name,
                    header["NUM_STATES"],
                )
            )

        states = [name for name in body if name != "HEADER"]
        expected_states = set(scenario.state_names())
        actual_states = set(states)
        if actual_states != expected_states:
            missing_states = tuple(sorted(expected_states - actual_states))
            extra_states = tuple(sorted(actual_states - expected_states))
            raise SMSLValidationError(
                "Scenario `{}` has state inventory mismatch: expected {} states, found {}.".format(
                    scenario.name,
                    scenario.state_count,
                    len(states),
                ),
                diagnostics=ValidationDiagnostics(
                    scenario_name=scenario.name,
                    expected_state_count=scenario.state_count,
                    actual_state_count=len(states),
                    missing_states=missing_states,
                    extra_states=extra_states,
                ),
            )

        for state_name in states:
            if not scenario.state_name_is_valid(state_name):
                raise SMSLValidationError("Invalid state name: {}".format(state_name))
            transitions = body[state_name]
            if not isinstance(transitions, dict):
                raise SMSLValidationError("Transitions for {} must be a dictionary.".format(state_name))
            for operation_name, next_state in transitions.items():
                if not scenario.operation_name_is_valid(operation_name):
                    raise SMSLValidationError("Invalid operation name: {}".format(operation_name))
                if not isinstance(next_state, str) or not scenario.state_name_is_valid(next_state):
                    raise SMSLValidationError("Invalid next state for {} in {}.".format(operation_name, state_name))

    @staticmethod
    def validate_matches_ground_truth(smsl: Dict[str, Any], scenario_name: str) -> None:
        expected_path = GROUND_TRUTH_DIR / "{}.json".format(scenario_name)
        if not expected_path.exists():
            raise SMSLValidationError("Ground-truth SMSL file is missing: {}".format(expected_path))

        expected_smsl = read_json(str(expected_path))
        if smsl == expected_smsl:
            return

        difference = SMSLValidator._describe_first_difference(
            actual=smsl,
            expected=expected_smsl,
            path="$",
        )
        raise SMSLValidationError(
            "Generated SMSL does not match ground truth `{}`. {}".format(
                expected_path,
                difference,
            )
        )

    @staticmethod
    def _describe_first_difference(actual: Any, expected: Any, path: str) -> str:
        if type(actual) is not type(expected):
            return "Type mismatch at {}: generated {} but expected {}.".format(
                path,
                type(actual).__name__,
                type(expected).__name__,
            )

        if isinstance(expected, dict):
            actual_keys = set(actual.keys())
            expected_keys = set(expected.keys())
            missing_keys = sorted(expected_keys - actual_keys)
            extra_keys = sorted(actual_keys - expected_keys)
            if missing_keys:
                return "Missing key at {}: expected `{}`.".format(path, missing_keys[0])
            if extra_keys:
                return "Unexpected key at {}: generated `{}`.".format(path, extra_keys[0])
            for key in expected:
                child_path = "{}.{}".format(path, key) if path != "$" else "$.{}".format(key)
                if actual[key] != expected[key]:
                    return SMSLValidator._describe_first_difference(actual[key], expected[key], child_path)
            return "Mismatch detected at {}.".format(path)

        if isinstance(expected, list):
            if len(actual) != len(expected):
                return "List length mismatch at {}: generated {} but expected {}.".format(
                    path,
                    len(actual),
                    len(expected),
                )
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                if actual_item != expected_item:
                    return SMSLValidator._describe_first_difference(
                        actual_item,
                        expected_item,
                        "{}[{}]".format(path, index),
                    )
            return "Mismatch detected at {}.".format(path)

        return "Value mismatch at {}: generated {!r} but expected {!r}.".format(
            path,
            actual,
            expected,
        )

    @staticmethod
    def validate_phase_shape(smsl: Dict[str, Any], scenario_name: str, phase_index: int) -> None:
        if phase_index not in {1, 2, 3}:
            raise ValueError("Intermediate phase validation only supports phases 1, 2, and 3.")
        if not isinstance(smsl, dict) or set(smsl.keys()) != {"SB1"}:
            raise SMSLValidationError("Phase {} JSON must have exactly one top-level key: `SB1`.".format(phase_index))

        body = smsl["SB1"]
        if not isinstance(body, dict) or "HEADER" not in body:
            raise SMSLValidationError("Phase {} JSON must contain `SB1.HEADER`.".format(phase_index))

        scenario = get_scenario_spec(scenario_name)
        header = body["HEADER"]
        required_header = {"INITIAL", "NUM_FACTS", "NUM_STATES", "SUB_SBS"}
        if not isinstance(header, dict) or set(header.keys()) != required_header:
            raise SMSLValidationError(
                "Phase {} `HEADER` must contain `INITIAL`, `NUM_FACTS`, `NUM_STATES`, and `SUB_SBS`.".format(
                    phase_index
                )
            )

        if not isinstance(header["INITIAL"], str) or not scenario.phase_state_name_is_valid(header["INITIAL"]):
            raise SMSLValidationError("Phase {} `HEADER.INITIAL` must be a symbolic `State_...` string.".format(phase_index))
        if not isinstance(header["NUM_FACTS"], int):
            raise SMSLValidationError("Phase {} `HEADER.NUM_FACTS` must be an integer.".format(phase_index))
        if not isinstance(header["NUM_STATES"], int):
            raise SMSLValidationError("Phase {} `HEADER.NUM_STATES` must be an integer.".format(phase_index))
        if not isinstance(header["SUB_SBS"], list):
            raise SMSLValidationError("Phase {} `HEADER.SUB_SBS` must be a list.".format(phase_index))

        for state_name, transitions in body.items():
            if state_name == "HEADER":
                continue
            if not scenario.phase_state_name_is_valid(state_name):
                raise SMSLValidationError(
                    "Phase {} contains an invalid symbolic state name: {}".format(phase_index, state_name)
                )
            if not isinstance(transitions, dict):
                raise SMSLValidationError(
                    "Phase {} transitions for {} must be a dictionary.".format(phase_index, state_name)
                )
            for operation_name, next_state in transitions.items():
                if not scenario.phase_operation_name_is_valid(operation_name):
                    raise SMSLValidationError(
                        "Phase {} contains an invalid symbolic operation name: {}".format(
                            phase_index,
                            operation_name,
                        )
                    )
                if not isinstance(next_state, str) or not scenario.phase_state_name_is_valid(next_state):
                    raise SMSLValidationError(
                        "Phase {} has an invalid next state for {} in {}.".format(
                            phase_index,
                            operation_name,
                            state_name,
                        )
                    )
