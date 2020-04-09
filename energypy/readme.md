# API driven development

## Setup

```bash
pip install energypy
```

## Usage

To train an agent:

```bash
energypy train
	--agent random
	--env battery
	--cpu 4
```

To test an agent:

```bash
energypy test
	--agent ppo
	--env battery
```

## Philosophy

- specalized, not general

- distributed (`ray`)
- single learner, multiple data collectors

- json everywhere
- challenge = can't seralize np arrays
- solution = hold shape infomation on agents / envs


RL
- as few algorithms as possible
- off policy
- continuous action space
- single agent (no multiagent)
- flexibility on function approximation
