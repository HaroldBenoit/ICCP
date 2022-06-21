---
layout: default
title: 2. First steps
parent: Get Started
nav_order: 2
permalink: /docs/get-started/first_steps
---

# First steps

## Introduction

This tutorial serves the purpose of getting a first feel of how `ICCP` works and just getting started with how agent, environment, and loggers interact with each other.

In this tutorial:

- We import the relevant packages.
- We instantiate an environment.
- We instantiate an agent.
- Train the agent and log the results.


## Code

```python
from environment.DiscreteEnvironment import DiscreteSimpleEnvironment
from agent.DQN_Agent import DQNAgent

env = DiscreteSimpleEnvironment()
agent = DQNAgent(env)

## please put the absolute path to the ICCP project
logging_path = r"{absolute path to ICCP}\logs\simple_simulation"
results_path, summary_df = agent.train(logging_path= logging_path, num_episodes=1)
```
