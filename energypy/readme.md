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

Agents 
- take actions
- generate transitions

Learners change parameters
- generate new parameters

energypy conclusions
- platform = the cloud

important data structures
- transitions (transition server)
- parameters (parameter servers)

## Philosophy

- specalized, not general

- distributed (`ray`)
- single learner, multiple data collectors

- json everywhere
- challenge = can't seralize np arrays
- solution = hold shape infomation on agents / envs

- no inheritance

- use f strings

RL
- as few algorithms as possible (value of the library = the choice of algorithm)
- off policy
- continuous action space
- single agent (no multiagent)
- flexibility on function approximation

## Data oriented design - [Mike Acton CppCon 2014](https://www.youtube.com/watch?v=rX0ItVEVjHc)

All about data transformations

Lies of OOP
1. software is a platform
- hardware is the platform
- hardware drives the solution

2. code designed around a model of the world
- world models hide data

Confusing two problems
1. maintenance (changes to access of data)
2. understanding properties of the data (solving the problem)

We often sacrifice the ease of changing access

In real life, classes of chairs are similar
- but in data transformations, these real world classes are only superficially similar
- handling & transformation of data is very different
- world models attempt to idealize the problem

3. code more important than the data
- only purpose of code == to transform data
- problem = to transform data
- writing code is only the tool!
- only write code that transforms data in a meaningful way
- only solve for data transformations

Cannot future proof code bases 
- it will have to change

Solve for the most common case 
- not the most generic

