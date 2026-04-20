# Verified Phase Scripts

This folder contains handwritten four-phase pipelines that reproduce the checked-in SMSL JSON files.

Phase contract:
- Phase 1 enumerates the full symbolic state space and all candidate symbolic operations under each state.
- Phase 2 filters states that violate the rules.
- Phase 3 filters operations that are not possible under their source state.
- Phase 4 applies each surviving operation, checks whether the resulting state matches the carried target state, repairs or removes mismatches, and removes invalid transitions.

Scenarios:
- `hanoi/`
- `river_crossing/`
- `chess/`

Each scenario folder contains:
- `phase_1.py`
- `phase_2.py`
- `phase_3.py`
- `phase_4.py`

Run one scenario end to end:

```bash
python smsl_generation/verified_phases/run_pipeline.py hanoi --check
python smsl_generation/verified_phases/run_pipeline.py river_crossing --check
python smsl_generation/verified_phases/run_pipeline.py chess --check
```

This writes the intermediate JSON files into:

```text
smsl_generation/verified_phases/output/<scenario>/
```

`--check` compares the produced `phase_4.json` against `smsl_gensim/SMSL_JSON/<scenario>.json`.
