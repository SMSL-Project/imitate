# SMSL GenSim

`smsl_gensim/` provides the GenSim and CLIPort simulation backend for language-conditioned robotic manipulation experiments.

This package supports the standard GenSim workflow:

- generate robotic simulation tasks with language-model prompts
- register generated CLIPort task code
- collect demonstrations from scripted oracle policies
- train CLIPort imitation-learning policies
- evaluate trained checkpoints

For SMSL JSON generation, use the sibling package `../smsl_generation/`.

## Layout

```text
smsl_gensim/
  gensim/              Task generation pipeline
  cliport/             Simulation environment, tasks, agents, configs, train/eval
  cliport/generated_tasks/
                       Active SMSL primitive task modules
  cliport/generated_tasks_backup/
                       Backup of the copied generated task modules
  prompts/             Prompt templates and task-memory JSON files
  smsl_gensim/         SMSL scenario files and scenario inspection commands
  scripts/             Dataset and policy training helpers
  requirements.txt     Python dependencies
  setup.py             Editable install entry point
```

## Installation

From `imitate/smsl_gensim`, create the SMSL GenSim Conda environment:

```bash
conda create -n smsl-gensim python=3.8 -y
conda activate smsl-gensim
pip install --upgrade pip
pip install -r requirements.txt
python setup.py develop
cp .env.example .env
export GENSIM_ROOT="$(pwd)"
```

Set the OpenAI key in `.env`:

```bash
OPENAI_API_KEY=YOUR_KEY
```

Do not reuse the SMSL generation environment for this package. `smsl_gensim` uses simulation, robotics, training, and evaluation dependencies.

`GENSIM_ROOT` is used by the Hydra configs to resolve data, asset, output, and checkpoint paths.

## Generate SMSL Task Code

Generate CLIPort primitive task modules from an SMSL scenario:

```bash
python -m smsl_gensim generate-tasks hanoi --backend llm
```

The command reads `smsl_gensim/SMSL_JSON/<scenario>.json`, asks the language model for a task specification, asks it to write one CLIPort task module from that specification and any requested URDF inspection, writes the primitive task list to `smsl_gensim/TASK_LIST/<scenario>.json`, and writes task code to `cliport/generated_tasks/`.

The LLM prompts are in `prompts/smsl_task_generation_prompt/`.

To regenerate existing task files with the language model:

```bash
python -m smsl_gensim generate-tasks hanoi --backend llm --overwrite
python -m smsl_gensim generate-tasks chess --backend llm --overwrite
python -m smsl_gensim generate-tasks river_crossing --backend llm --overwrite
```

The deterministic task generator is available only when explicitly requested:

```bash
python -m smsl_gensim generate-tasks hanoi --backend template --overwrite
```

## Inspect SMSL Scenarios

List available SMSL scenarios:

```bash
python -m smsl_gensim list
```

Show a scenario summary:

```bash
python -m smsl_gensim show hanoi
```

List primitive tasks for a scenario:

```bash
python -m smsl_gensim tasks hanoi
```

List constrained SMSL states for a scenario:

```bash
python -m smsl_gensim states hanoi
```

Check that a scenario has its primitive task modules, registry names, and assets:

```bash
python -m smsl_gensim check-tasks hanoi
```

Check one constrained scene:

```bash
python -m smsl_gensim check-scene hanoi State_aaa
```

Check every constrained scene for a scenario:

```bash
python -m smsl_gensim check-scenes hanoi
```

Check the SMSL demo collection plan:

```bash
python -m smsl_gensim demos-plan hanoi
```

This treats every SMSL transition as an independent demo target. For example, `hanoi` has 78 transition targets even though it has 9 primitive task names.

Show the full pipeline status for a scenario:

```bash
python -m smsl_gensim status hanoi
```

Here `--n 1` means one planned demo for each SMSL transition.

Collect one demo for every SMSL transition in a scenario:

```bash
python -m smsl_gensim collect-demos hanoi --n 1 --mode train
```

Check the collected dataset before training:

```bash
python -m smsl_gensim check-data hanoi --n 1 --mode train
```

The checker verifies the expected transition count, saved episode files, and SMSL scene metadata.


Fill only the missing demos reported by `check-data`:

```bash
python -m smsl_gensim collect-missing hanoi --n 1 --mode train
```

This is useful after a long collection run where only a few transitions failed. It does not rerun transitions that already have the requested number of demos.

Merge transition demos by primitive operation:

```bash
python -m smsl_gensim merge-data hanoi --mode train
```

The merged dataset groups all demos with the same primitive task name under `data/smsl_operations/<scenario>/<mode>/`. The generated `index.json` files keep the original start state, operation, end state, and source transition directory for each episode.

To remove the transition-level source split after the merge succeeds:

```bash
python -m smsl_gensim merge-data hanoi --mode train --remove-source
```

Train one merged operation policy:

```bash
python -m smsl_gensim train-operation hanoi move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand
```

By default, this uses all merged training episodes for that operation and one validation episode from the same split.

If a separate validation split has been collected and merged, use it with `--val-mode val`.

Set training parameters with command-line flags:

```bash
python -m smsl_gensim train-operation hanoi \
  move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand \
  --batch-size 2 \
  --training-step-scale 200 \
  --gpu 0
```

Common parameters:

- `--batch-size`: mini-batch size. CLIPort's default is `8`.
- `--n`: number of training demos to use. By default, all merged episodes for the operation are used.
- `--n-val`: number of validation demos. The default is `1`.
- `--n-steps`: CLIPort step target. steps per epoch = n_demos / batch_size
- `--training-step-scale`: max-epoch override. The CLIPort config default is `200`, so `--n-steps` only controls training length when this is set to `-1`.
- `--n-rotations`: number of rotation bins. CLIPort's default is `36`.
- `--gpu`: GPU index, such as `0`. Use `-1` for the configured default.
- `--no-cache`: read episodes from disk instead of caching the dataset in memory.
- `--debug`: run PyTorch Lightning's fast development pass.
- `--log`: enable Weights & Biases logging.
- `--exp-dir`: experiment output directory. The default is `exps`.

To train by step count instead of the epoch scale:

```bash
python -m smsl_gensim train-operation hanoi \
  move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand \
  --batch-size 2 \
  --training-step-scale -1 \
  --n-steps 10000
```

Advanced CLIPort hyperparameters, such as `train.lr`, are stored in `cliport/cfg/train.yaml`. 

Train every operation policy in a scenario:

```bash
python -m smsl_gensim train-scenario hanoi
```

This trains one operation model at a time.
To resume an interrupted scenario training run, add `--skip-existing`.

Evaluate one trained operation policy on its constrained SMSL start states:

```bash
python -m smsl_gensim eval-operation hanoi move-gray-hanoi-ring-to-the-center-of-blue-hanoi-stand
```

The result is saved under `evals/smsl_operations/<scenario>/<mode>/`.

Evaluate every trained operation policy in a scenario:

```bash
python -m smsl_gensim eval-scenario hanoi
```

This writes per-operation results and a scenario summary.
To resume an interrupted scenario evaluation run, add `--skip-existing`.

Collect one transition:

```bash
python -m smsl_gensim collect-one hanoi \
  --from-state State_aaa \
  --operation move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand
```

By default, collection runs headless and does not open the PyBullet simulator window. To watch the simulator, add `--disp`:

```bash
python -m smsl_gensim collect-one hanoi \
  --from-state State_aaa \
  --operation move_gray_hanoi_ring_to_the_center_of_blue_hanoi_stand \
  --disp
```

The same display flag is available for full scenario collection:

```bash
python -m smsl_gensim collect-demos hanoi --n 1 --mode train --disp
```

The included SMSL primitive tasks cover:

- `chess`
- `hanoi`
- `river_crossing`

## Generate General GenSim Tasks

The original GenSim prompt pipeline is still included under `gensim/` and `prompts/`. SMSL scenarios should use `python -m smsl_gensim generate-tasks <scenario>` so the generated task list matches the SMSL transition graph.

## Collect Demonstrations

Use CLIPort's scripted oracle to generate demonstrations for a task:

```bash
python cliport/demos.py \
  n=10 \
  task=packing-boxes-pairs-seen-colors \
  mode=train \
  disp=False
```

Datasets are saved under `data/<task>-<split>/`.

## Train Policies

Train a CLIPort policy from collected demonstrations:

```bash
python cliport/train.py \
  train.task=packing-boxes-pairs-seen-colors \
  train.agent=cliport \
  train.n_demos=10 \
  train.n_steps=1000
```

Experiment outputs are saved under `exps/`.

For SMSL operation datasets, prefer `python -m smsl_gensim train-operation ...`; it sets the merged dataset paths for CLIPort.

## Evaluate Checkpoints

Evaluate a trained checkpoint with the CLIPort evaluation entry point:

```bash
python cliport/eval.py \
  eval_task=packing-boxes-pairs-seen-colors \
  agent=cliport \
  mode=val \
  n_demos=10
```

Evaluation results are written to the checkpoint directory as JSON.

## Generated Files

The repository ignores runtime outputs such as:

- `data/`
- `exps/`
- `evals/`
- `output/`
- `outputs/`
- `results/`
- `lightning_logs/`
- `wandb/`
- checkpoints and videos

The prompt-memory files in `prompts/data/*.json` are source inputs and should remain version controlled.

## Attribution

This package is based on and modified from GenSim:

- Project: https://github.com/liruiw/GenSim
- Paper page: https://liruiw.github.io/gensim
- Original authors: Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, Xiaolong Wang

The work's citation is below

```bibtex
@inproceedings{wang2024gensim,
  title={Gensim: Generating robotic simulation tasks via large language models},
  author={Wang, Lirui and Ling, Yiyang and Yuan, Zhecheng and Shridhar, Mohit and Bao, Chen and Qin, Yuzhe and Wang, Bailin and Xu, Huazhe and Wang, Xiaolong},
  booktitle={International Conference on Learning Representations},
  volume={2024},
  pages={4890--4924},
  year={2024}
}
```

## Citations

If you find this work useful, please consider citing:

```bibtex
@inproceedings{mu2025look,
  title={Look before you leap: Using serialized state machine for language conditioned robotic manipulation},
  author={Mu, Tong and Liu, Yihao and Armand, Mehran},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={8096--8102},
  year={2025},
  organization={IEEE}
}
```
