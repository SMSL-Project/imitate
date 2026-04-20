from __future__ import annotations

import argparse
from typing import Optional, Sequence

from smsl_generation.app.config import load_generation_config, render_generation_config
from smsl_generation.app.generator import build_generator, default_artifact_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario-specific SMSL JSON from a YAML config file."
    )
    parser.add_argument("config", help="Path to a generation YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Load and print the resolved config without calling the model.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_generation_config(args.config)
    if args.dry_run:
        print(render_generation_config(config).strip())
        return

    generator = build_generator(
        config=config,
        artifact_dir=default_artifact_dir(config.scenario, config.mode) if not config.artifact_dir else config.artifact_dir,
    )
    result = generator.generate()

    print("Config:", args.config)
    print("Artifact directory:", result.artifact_dir)
    print("Prompt:", result.prompt_path)
    print("Raw response:", result.response_path)
    print("Generated SMSL:", result.generated_json_path)
    if result.published_json_path:
        print("Published SMSL:", result.published_json_path)
    for phase_script_path in result.phase_script_paths:
        print("Phase script:", phase_script_path)
