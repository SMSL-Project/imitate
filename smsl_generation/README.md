# SMSL Generation Package

`smsl_generation/` generates SMSL JSON artifacts intended to match the scenario files in `smsl_gensim/smsl_gensim/SMSL_JSON/`.

## Package Layout

```text
smsl_generation/
  __main__.py                                Entry point for `python -m smsl_generation`
  .env.example                               Example local environment file for API keys
  README.md                                  Package guide and usage notes
  requirements.txt                           Runtime dependency list for this package
  app/
    __init__.py                              Public app exports
    cli.py                                   Loads one config file and starts generation
    config.py                                Config schema and YAML loading
    generator.py                             Prompt rendering, model call, parsing, validation, execution
  configs/
    {scenario}_{mode}.yaml                   Simple configs (e.g. hanoi_indirect.yaml)
    {scenario}_{mode}_{style}_{model}.yaml   Full configs (e.g. hanoi_indirect_detailed_gemini31pro.yaml)
  prompts/
    direct/
      {style}/{scenario}/direct.txt          Direct prompt per style and scenario
    indirect/
      {style}/{scenario}/indirect.txt        Indirect prompt per style and scenario
  support/
    __init__.py                              Support exports
    assets.py                                Prompt and template loading
    client.py                                Multi-provider LLM client (OpenAI, Gemini, Claude)
    env.py                                   .env file loader
    files.py                                 File read/write helpers
    parser.py                                Model-response parsing helpers
    scenarios.py                             Scenario-specific naming and asset locations
    validation.py                            SMSL shape validation using the selected scenario
  templates/
    {scenario}/
      smsl_template.json                     Structural JSON template with placeholders
    {style}/
      common/
        phase_1.py … phase_4.py              Shared fill-in scaffolds for each phase
```

## Quick Start

Create a fresh Conda environment, install the package requirements, and store your API keys in the package-local `.env` file:

```bash
conda create -n smsl-generation python=3.11 -y
conda activate smsl-generation
pip install -r smsl_generation/requirements.txt
cp smsl_generation/.env.example smsl_generation/.env
```

Then edit `smsl_generation/.env` and set the key(s) for the provider(s) you plan to use:

```bash
OPENAI_API_KEY=YOUR_KEY
GEMINI_API_KEY=YOUR_KEY
ANTHROPIC_API_KEY=YOUR_KEY
```

Validate a config without calling the model:

```bash
python -m smsl_generation smsl_generation/configs/hanoi_indirect.yaml --dry-run
```

Run indirect generation (recommended) with the default model (`gpt-4o`):

```bash
python -m smsl_generation smsl_generation/configs/hanoi_indirect.yaml
python -m smsl_generation smsl_generation/configs/river_crossing_indirect.yaml
python -m smsl_generation smsl_generation/configs/chess_indirect.yaml
```

Run with a specific model and style (example using Gemini):

```bash
python -m smsl_generation smsl_generation/configs/hanoi_indirect_detailed_gemini31pro.yaml
```

In indirect mode, the package makes four separate model calls, one per phase script. It then executes those scripts locally to produce the final SMSL JSON.
The package does not ship puzzle-specific answer scripts for the scenarios; it only ships shared phase scaffolds plus scenario prompts and validators.

Run direct generation only if you specifically want the one-shot path:

```bash
python -m smsl_generation smsl_generation/configs/hanoi_direct.yaml
python -m smsl_generation smsl_generation/configs/river_crossing_direct.yaml
python -m smsl_generation smsl_generation/configs/chess_direct.yaml
```

The generated artifacts are written to the configured `artifact_dir`, and the published SMSL JSON goes to `out_json`.

## Config Fields

Each config file supports these fields:

```yaml
scenario: hanoi
mode: indirect
model: gpt-4o
provider: auto
temperature: 0.0
artifact_dir: smsl_generation/output/hanoi_indirect
out_json: smsl_generation/output/hanoi_indirect.json
execute_phase_scripts: true
validation_retries: 0
style: detailed
```

- `scenario`
  One of `hanoi`, `river_crossing`, or `chess`.

- `mode`
  `indirect` is the recommended default. It asks the model for one phase script at a time across four separate calls, then executes them locally to produce the final SMSL JSON.
  In phases 1, 2, and 3 those scripts use symbolic `State_...` and `Op_...` names.
  In phase 4 the script maps symbolic operations to repo task names and emits final next states.
  `direct` asks the model for the final SMSL JSON in one shot, but it is less reliable.

- `model`
  Model name. Supports any model from the supported providers. See the table below for examples.

- `provider`
  One of `auto`, `openai`, `gemini`, or `claude`. Default is `auto`, which infers the provider from the model name prefix (e.g. `gpt-*` / `o*` → OpenAI, `gemini-*` → Gemini, `claude-*` → Claude). Set explicitly if auto-detection does not cover your model name.

- `style`
  One of `detailed`, `concise`, or `example_driven`. Default is `detailed`.
  Controls both the prompt wording and the phase scaffold templates used during generation.
  Each style has its own set of prompts under `prompts/{direct,indirect}/{style}/` and phase scaffolds under `templates/{style}/common/`.

- `temperature`
  Sampling temperature for the model call.

- `artifact_dir`
  Folder where the rendered prompt, raw model response, and generated phase scripts are written.

- `out_json`
  Final published SMSL JSON path.

- `execute_phase_scripts`
  Required in `indirect`.
  The final SMSL JSON is generated by executing the returned phase scripts.

- `validation_retries`
  Applies only to `direct`.
  The default is `5`.
  When direct validation finds a missing or extra state inventory mismatch, the generator warns and asks the model to regenerate with stronger self-checking.
  In `indirect`, the package executes one returned script set and stops on validation failure.

## Supported Providers

| Provider  | SDK Package    | API Key Env Var    | Example Models                                        |
|-----------|----------------|--------------------|-------------------------------------------------------|
| `openai`  | `openai`       | `OPENAI_API_KEY`   | `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini`             |
| `gemini`  | `google-genai` | `GEMINI_API_KEY`   | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`    |
| `claude`  | `anthropic`    | `ANTHROPIC_API_KEY`| `claude-opus-4-6`, `claude-haiku-4-5-20251001`        |
