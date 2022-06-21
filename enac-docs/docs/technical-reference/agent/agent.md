---
layout: default
title: Agent
parent: Technical reference
nav_order: 2
permalink: /docs/technical-reference/agent
has_children: true
---

# Agents

`ICCP` contains a folder called `agent` where all the logic and code about reinforcement learning agents may be found. An example is given below as to how you could import elements from `agent`.

```python
# main abstract class
from agent.Agent import Agent
# concrete implementation of a DQN agent
from agent.DQN_Agent import DQNAgent
```






