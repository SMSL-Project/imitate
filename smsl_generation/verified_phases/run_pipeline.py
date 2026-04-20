from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIED_DIR = ROOT / "smsl_generation" / "verified_phases"
EXPECTED_DIR = ROOT / "smsl_gensim" / "smsl_gensim" / "SMSL_JSON"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("hanoi", "river_crossing", "chess"))
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where the intermediate phase JSON files should be written.",
    )
    parser.add_argument("--check", action="store_true", help="Compare phase_4.json to smsl_gensim/SMSL_JSON/<scenario>.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = VERIFIED_DIR / args.scenario
    output_dir = Path(args.output_dir) if args.output_dir else VERIFIED_DIR / "output" / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            str(scenario_dir / "phase_1.py"),
            "--output",
            str(output_dir / "phase_1.json"),
        ],
        [
            sys.executable,
            str(scenario_dir / "phase_2.py"),
            "--input",
            str(output_dir / "phase_1.json"),
            "--output",
            str(output_dir / "phase_2.json"),
        ],
        [
            sys.executable,
            str(scenario_dir / "phase_3.py"),
            "--input",
            str(output_dir / "phase_2.json"),
            "--output",
            str(output_dir / "phase_3.json"),
        ],
        [
            sys.executable,
            str(scenario_dir / "phase_4.py"),
            "--input",
            str(output_dir / "phase_3.json"),
            "--output",
            str(output_dir / "phase_4.json"),
        ],
    ]

    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT)

    print("Wrote:")
    for phase_index in range(1, 5):
        print(output_dir / "phase_{}.json".format(phase_index))

    if args.check:
        with open(output_dir / "phase_4.json", "r", encoding="utf-8") as fh:
            generated = json.load(fh)
        with open(EXPECTED_DIR / "{}.json".format(args.scenario), "r", encoding="utf-8") as fh:
            expected = json.load(fh)
        print("Matches expected:", generated == expected)


if __name__ == "__main__":
    main()
