from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from smsl_generation.app.config import GenerationConfig
from smsl_generation.support.assets import GENERATION_DIR, PromptAssets
from smsl_generation.support.client import ChatClient, create_chat_client
from smsl_generation.support.files import read_json, write_json, write_text
from smsl_generation.support.parser import extract_json_from_text, extract_single_python_block
from smsl_generation.support.validation import SMSLValidationError, SMSLValidator, ValidationDiagnostics


SYSTEM_PROMPTS = {
    "direct": (
        "Follow the user prompt exactly and return only one fenced json block. "
        "Do not add prose before or after the block."
    ),
    "indirect": (
        "Follow the user prompt exactly and return exactly one fenced python block with no prose."
    ),
}

RETRY_GUIDANCE = (
    "The previous attempt failed validation. Recompute the full state inventory from scratch, "
    "carefully check missing and extra states, and verify completeness before answering. "
    "Reason step by step internally, but do not reveal your reasoning. Return only the required fenced block output."
)


@dataclass(frozen=True)
class GenerationResult:
    artifact_dir: str
    prompt_path: str
    response_path: str
    generated_json_path: str
    published_json_path: Optional[str]
    phase_script_paths: List[str]


class BaseSMSLGenerator(ABC):
    mode: str = ""
    system_prompt: str = ""
    uses_attempt_artifacts: bool = False

    def __init__(
        self,
        config: GenerationConfig,
        artifact_dir: str,
        client: Optional[ChatClient] = None,
    ) -> None:
        if config.mode != self.mode:
            raise ValueError(
                "{} requires config.mode=`{}`, but received `{}`.".format(
                    self.__class__.__name__,
                    self.mode,
                    config.mode,
                )
            )
        self.config = config
        self.client = client or create_chat_client(
            provider=config.provider, model=config.model, temperature=config.temperature,
        )
        self.assets = PromptAssets.load(config.scenario, style=config.style)
        self.artifact_dir = os.path.abspath(artifact_dir)
        self.out_json = os.path.abspath(config.out_json) if config.out_json else None
        self.execute_phase_scripts = config.execute_phase_scripts

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print("[smsl_generation] {}".format(message), file=sys.stderr)

    def _count_states(self, smsl: Dict[str, Any]) -> int:
        sb = smsl.get("SB1", {})
        return len([state_name for state_name in sb if state_name != "HEADER"])

    def _header_summary(self, smsl: Dict[str, Any]) -> str:
        sb = smsl.get("SB1", {})
        header = sb.get("HEADER", {})
        return "initial={}, header_num_states={}, actual_states={}".format(
            header.get("INITIAL", "unknown"),
            header.get("NUM_STATES", "unknown"),
            self._count_states(smsl),
        )

    def generate(self) -> GenerationResult:
        os.makedirs(self.artifact_dir, exist_ok=True)
        self._log(
            "Starting {} generation for scenario `{}` using model `{}`.".format(
                self.mode,
                self.config.scenario,
                self.config.model,
            )
        )
        self._log("Artifacts will be written to `{}`.".format(self.artifact_dir))

        base_prompt = self.build_prompt()
        prompt_path = os.path.join(self.artifact_dir, "prompt.txt")
        response_path = os.path.join(self.artifact_dir, "raw_response.txt")
        generated_json_path = os.path.join(self.artifact_dir, "generated_smsl.json")
        prompt = base_prompt

        phase_script_paths: List[str] = []
        for attempt_index in range(self.max_attempts()):
            self._log("Preparing {}.".format(self._format_attempt_label(attempt_index)))
            write_text(prompt_path, prompt)
            self._write_attempt_artifact("prompt", attempt_index, prompt)
            self._log("Prompt saved to `{}`.".format(prompt_path))

            self._log("Requesting model completion.")
            response = self.client.complete(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
            )
            write_text(response_path, response)
            self._write_attempt_artifact("raw_response", attempt_index, response)
            self._log("Raw response saved to `{}`.".format(response_path))

            phase_script_paths, final_smsl = self.extract_generation_output(response, attempt_index)
            self._log("Model response parsed successfully.")

            try:
                self._validate(final_smsl)
                self._log("Validation passed: {}.".format(self._header_summary(final_smsl)))
                break
            except SMSLValidationError as exc:
                retrying = self.should_retry(attempt_index, exc)
                stem = "validation_warning" if retrying else "validation_error"
                detail_path = self._detail_path(stem, attempt_index)
                detail_text = self._render_validation_message(exc, attempt_index, detail_path, retrying)
                write_text(detail_path, detail_text)
                print(detail_text, file=sys.stderr)
                if not retrying:
                    raise
                prompt = self.build_retry_prompt(base_prompt, exc.diagnostics)
        else:
            raise RuntimeError("Generation exhausted all validation retries without returning a valid SMSL.")

        write_json(generated_json_path, final_smsl)
        self._log("Generated SMSL written to `{}`.".format(generated_json_path))

        published_json_path = None
        if self.out_json:
            write_json(self.out_json, final_smsl)
            published_json_path = self.out_json
            self._log("Published SMSL written to `{}`.".format(self.out_json))

        return GenerationResult(
            artifact_dir=self.artifact_dir,
            prompt_path=prompt_path,
            response_path=response_path,
            generated_json_path=generated_json_path,
            published_json_path=published_json_path,
            phase_script_paths=phase_script_paths,
        )

    @abstractmethod
    def build_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_generation_output(self, response: str, attempt_index: int) -> tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError

    def max_attempts(self) -> int:
        return 1

    def should_retry(self, attempt_index: int, error: SMSLValidationError) -> bool:
        return False

    def build_retry_prompt(self, base_prompt: str, diagnostics: ValidationDiagnostics | None) -> str:
        lines = [base_prompt, "", RETRY_GUIDANCE]
        if diagnostics is not None:
            lines.extend(
                [
                    "",
                    "Validation warning:",
                    "Expected state count: {}".format(diagnostics.expected_state_count),
                    "Actual state count: {}".format(diagnostics.actual_state_count),
                    "Missing states: {}".format(self._format_state_list(diagnostics.missing_states)),
                    "Unexpected states: {}".format(self._format_state_list(diagnostics.extra_states)),
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _validate(self, smsl: Dict[str, Any]) -> None:
        SMSLValidator.validate_shape(smsl, self.config.scenario)
        SMSLValidator.validate_matches_ground_truth(smsl, self.config.scenario)

    def _format_state_list(self, states: Sequence[str], limit: int = 25) -> str:
        if not states:
            return "none"
        rendered = ", ".join(states[:limit])
        if len(states) > limit:
            rendered += ", ... ({} more)".format(len(states) - limit)
        return rendered

    def _format_attempt_label(self, attempt_index: int) -> str:
        return "attempt {} of {}".format(attempt_index + 1, self.max_attempts())

    def _render_validation_message(
        self,
        error: SMSLValidationError,
        attempt_index: int,
        detail_path: str,
        retrying: bool,
    ) -> str:
        heading = "Validation warning" if retrying else "Validation failed"
        lines = [
            "{} for scenario `{}` on {}.".format(
                heading,
                self.config.scenario,
                self._format_attempt_label(attempt_index),
            ),
            "Problem: {}".format(error),
        ]
        if error.diagnostics is not None:
            lines.extend(
                [
                    "Expected states: {}".format(error.diagnostics.expected_state_count),
                    "Actual states: {}".format(error.diagnostics.actual_state_count),
                    "Missing states: {}".format(self._format_state_list(error.diagnostics.missing_states)),
                    "Unexpected states: {}".format(self._format_state_list(error.diagnostics.extra_states)),
                ]
            )
        lines.append("Details saved to: {}".format(detail_path))
        if retrying:
            lines.append("Action: retrying with a targeted prompt.")
        else:
            lines.append("Action: stopping because no validation retries remain.")
        return "\n".join(lines)

    def _run_phase_scripts(self, phase_script_paths: Sequence[str], attempt_index: int) -> Dict[str, Any]:
        outputs = [
            os.path.join(self.artifact_dir, "phase_1.json"),
            os.path.join(self.artifact_dir, "phase_2.json"),
            os.path.join(self.artifact_dir, "phase_3.json"),
            os.path.join(self.artifact_dir, "phase_4.json"),
        ]

        commands = [
            [sys.executable, phase_script_paths[0], "--output", outputs[0]],
            [sys.executable, phase_script_paths[1], "--input", outputs[0], "--output", outputs[1]],
            [sys.executable, phase_script_paths[2], "--input", outputs[1], "--output", outputs[2]],
            [sys.executable, phase_script_paths[3], "--input", outputs[2], "--output", outputs[3]],
        ]

        for phase_index, command in enumerate(commands, start=1):
            script_path = command[1] if len(command) > 1 else "unknown"
            self._log(
                "Executing phase {} script `{}`.".format(
                    phase_index,
                    script_path,
                )
            )
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    cwd=self.artifact_dir,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                detail_path = self._detail_path("phase_execution_error", attempt_index)
                detail_text = self._render_phase_execution_message(
                    attempt_index=attempt_index,
                    phase_index=phase_index,
                    command=command,
                    error=exc,
                    detail_path=detail_path,
                    output_path=outputs[phase_index - 1],
                )
                write_text(detail_path, detail_text)
                print(detail_text, file=sys.stderr)
                raise RuntimeError(
                    "Phase {} execution failed for scenario `{}`. Details saved to: {}".format(
                        phase_index,
                        self.config.scenario,
                        detail_path,
                    )
                ) from exc
            else:
                self._log(
                    "Phase {} script completed with exit status {}.".format(
                        phase_index,
                        completed.returncode,
                    )
                )
                if completed.stdout.strip():
                    self._log("Phase {} stdout:\n{}".format(phase_index, completed.stdout.strip()))
                if completed.stderr.strip():
                    self._log("Phase {} stderr:\n{}".format(phase_index, completed.stderr.strip()))

            phase_payload = read_json(outputs[phase_index - 1])
            self._log(
                "Phase {} output saved to `{}` with {}.".format(
                    phase_index,
                    outputs[phase_index - 1],
                    self._header_summary(phase_payload),
                )
            )
            if phase_index < 4:
                try:
                    SMSLValidator.validate_phase_shape(phase_payload, self.config.scenario, phase_index)
                except SMSLValidationError as exc:
                    detail_path = self._detail_path("phase_validation_error", attempt_index)
                    detail_text = self._render_phase_validation_message(
                        attempt_index=attempt_index,
                        phase_index=phase_index,
                        error=exc,
                        detail_path=detail_path,
                        output_path=outputs[phase_index - 1],
                    )
                    write_text(detail_path, detail_text)
                    print(detail_text, file=sys.stderr)
                    raise RuntimeError(
                        "Phase {} validation failed for scenario `{}`. Details saved to: {}".format(
                            phase_index,
                            self.config.scenario,
                            detail_path,
                        )
                    ) from exc
                else:
                    self._log("Phase {} symbolic contract validation passed.".format(phase_index))

        return read_json(outputs[-1])

    def _render_phase_execution_message(
        self,
        attempt_index: int,
        phase_index: int,
        command: Sequence[str],
        error: subprocess.CalledProcessError,
        detail_path: str,
        output_path: str,
    ) -> str:
        lines = [
            "Phase execution failed for scenario `{}` on {}.".format(
                self.config.scenario,
                self._format_attempt_label(attempt_index),
            ),
            "Problem: phase {} script exited with status {}.".format(phase_index, error.returncode),
            "Script: {}".format(command[1] if len(command) > 1 else "unknown"),
            "Output target: {}".format(output_path),
            "Details saved to: {}".format(detail_path),
            "Stdout:",
            error.stdout.strip() or "(empty)",
            "Stderr:",
            error.stderr.strip() or "(empty)",
        ]
        lines.append("Action: stopping because indirect mode does not retry.")
        return "\n".join(lines)

    def _render_phase_validation_message(
        self,
        attempt_index: int,
        phase_index: int,
        error: SMSLValidationError,
        detail_path: str,
        output_path: str,
    ) -> str:
        return "\n".join(
            [
                "Phase validation failed for scenario `{}` on {}.".format(
                    self.config.scenario,
                    self._format_attempt_label(attempt_index),
                ),
                "Problem: phase {} output does not satisfy the symbolic `State_...` / `Op_...` contract.".format(
                    phase_index
                ),
                "Details: {}".format(error),
                "Output target: {}".format(output_path),
                "Details saved to: {}".format(detail_path),
                "Action: stopping because indirect mode does not retry.",
            ]
        )

    def _detail_path(self, stem: str, attempt_index: int, suffix: str = ".txt") -> str:
        if self.uses_attempt_artifacts:
            return self._attempt_path(stem, attempt_index, suffix)
        return os.path.join(self.artifact_dir, "{}{}".format(stem, suffix))

    def _write_attempt_artifact(
        self,
        stem: str,
        attempt_index: int,
        content: str,
        suffix: str = ".txt",
    ) -> None:
        if not self.uses_attempt_artifacts:
            return
        write_text(self._attempt_path(stem, attempt_index, suffix), content)

    def _attempt_path(self, stem: str, attempt_index: int, suffix: str = ".txt") -> str:
        return os.path.join(self.artifact_dir, "{}_attempt_{}{}".format(stem, attempt_index + 1, suffix))


class DirectSMSLGenerator(BaseSMSLGenerator):
    mode = "direct"
    system_prompt = SYSTEM_PROMPTS["direct"]
    uses_attempt_artifacts = True

    def build_prompt(self) -> str:
        return self.assets.build_direct_prompt()

    def extract_generation_output(self, response: str, attempt_index: int) -> tuple[List[str], Dict[str, Any]]:
        del attempt_index
        return [], extract_json_from_text(response)

    def max_attempts(self) -> int:
        return self.config.validation_retries + 1

    def should_retry(self, attempt_index: int, error: SMSLValidationError) -> bool:
        return attempt_index < self.config.validation_retries and error.diagnostics is not None


class IndirectSMSLGenerator(BaseSMSLGenerator):
    mode = "indirect"
    system_prompt = SYSTEM_PROMPTS["indirect"]

    def build_prompt(self) -> str:
        return self.assets.build_indirect_phase_prompt(1)

    def extract_generation_output(self, response: str, attempt_index: int) -> tuple[List[str], Dict[str, Any]]:
        del response, attempt_index
        raise NotImplementedError("IndirectSMSLGenerator.generate() handles phase-by-phase generation.")

    def generate(self) -> GenerationResult:
        os.makedirs(self.artifact_dir, exist_ok=True)
        self._log(
            "Starting indirect generation for scenario `{}` using model `{}`.".format(
                self.config.scenario,
                self.config.model,
            )
        )
        self._log("Artifacts will be written to `{}`.".format(self.artifact_dir))

        prompt_path = os.path.join(self.artifact_dir, "prompt.txt")
        response_path = os.path.join(self.artifact_dir, "raw_response.txt")
        generated_json_path = os.path.join(self.artifact_dir, "generated_smsl.json")

        phase_script_paths: List[str] = []
        prompt_sections: List[str] = []
        response_sections: List[str] = []

        for phase_index in range(1, 5):
            phase_prompt = self.assets.build_indirect_phase_prompt(phase_index)
            self._log("Building prompt for phase {}.".format(phase_index))
            prompt_sections.append(self._render_phase_log_section("Phase {}".format(phase_index), phase_prompt))
            phase_prompt_path = os.path.join(self.artifact_dir, "phase_{}_prompt.txt".format(phase_index))
            write_text(phase_prompt_path, phase_prompt)
            self._log("Phase {} prompt saved to `{}`.".format(phase_index, phase_prompt_path))

            self._log("Requesting model completion for phase {}.".format(phase_index))
            response = self.client.complete(
                system_prompt=self.system_prompt,
                user_prompt=phase_prompt,
            )
            response_sections.append(self._render_phase_log_section("Phase {}".format(phase_index), response))
            phase_response_path = os.path.join(self.artifact_dir, "phase_{}_raw_response.txt".format(phase_index))
            write_text(phase_response_path, response)
            self._log("Phase {} raw response saved to `{}`.".format(phase_index, phase_response_path))

            phase_script_path = self._write_single_phase_script(response, phase_index)
            phase_script_paths.append(phase_script_path)
            self._log("Phase {} script extracted to `{}`.".format(phase_index, phase_script_path))

        write_text(prompt_path, "\n\n".join(prompt_sections).strip() + "\n")
        write_text(response_path, "\n\n".join(response_sections).strip() + "\n")
        self._log("Combined prompt log written to `{}`.".format(prompt_path))
        self._log("Combined response log written to `{}`.".format(response_path))

        self._log("Executing generated phase scripts.")
        final_smsl = self._run_phase_scripts(phase_script_paths, 0)
        try:
            self._validate(final_smsl)
        except SMSLValidationError as exc:
            detail_path = self._detail_path("validation_error", 0)
            detail_text = self._render_validation_message(exc, 0, detail_path, retrying=False)
            write_text(detail_path, detail_text)
            print(detail_text, file=sys.stderr)
            raise
        else:
            self._log("Final validation passed: {}.".format(self._header_summary(final_smsl)))

        write_json(generated_json_path, final_smsl)
        self._log("Generated SMSL written to `{}`.".format(generated_json_path))

        published_json_path = None
        if self.out_json:
            write_json(self.out_json, final_smsl)
            published_json_path = self.out_json
            self._log("Published SMSL written to `{}`.".format(self.out_json))

        return GenerationResult(
            artifact_dir=self.artifact_dir,
            prompt_path=prompt_path,
            response_path=response_path,
            generated_json_path=generated_json_path,
            published_json_path=published_json_path,
            phase_script_paths=phase_script_paths,
        )

    def _write_single_phase_script(self, response: str, phase_index: int) -> str:
        content = extract_single_python_block(response)
        path = os.path.join(self.artifact_dir, "phase_{}.py".format(phase_index))
        write_text(path, content + "\n")
        return path

    def _render_phase_log_section(self, heading: str, content: str) -> str:
        return "### {}\n{}".format(heading, content.rstrip())


def build_generator(
    config: GenerationConfig,
    artifact_dir: str,
    client: Optional[ChatClient] = None,
) -> BaseSMSLGenerator:
    if config.mode == "direct":
        return DirectSMSLGenerator(config=config, artifact_dir=artifact_dir, client=client)
    if config.mode == "indirect":
        return IndirectSMSLGenerator(config=config, artifact_dir=artifact_dir, client=client)
    raise ValueError("Unsupported generation mode: {}".format(config.mode))


class AgenticSMSLGenerator:
    def __new__(
        cls,
        config: GenerationConfig,
        artifact_dir: str,
        client: Optional[ChatClient] = None,
    ) -> BaseSMSLGenerator:
        return build_generator(config=config, artifact_dir=artifact_dir, client=client)


def default_artifact_dir(scenario: str, mode: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(GENERATION_DIR, "output", "{}_{}_{}".format(scenario, mode, stamp))
