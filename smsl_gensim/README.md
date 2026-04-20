# Look Before You Leap: Using Serialized State Machine for Language Conditioned Robotic Manipulation 

This sub-repository uses two previous projects:

- CLIPort: https://github.com/cliport/cliport
- GenSim: https://github.com/liruiw/GenSim

The core idea is:

1. Use symbolic task definitions (SMSL) to describe complex multi-step tasks as state transitions.
2. Generate/prepare task implementations and environment initializers from SMSL.
3. Collect demonstrations using those SMSL-driven task definitions.
4. Train GenSim and CLIPort-style imitation learning policies.
5. Evaluate checkpoints systematically across tasks and scenarios.

Coming soon.