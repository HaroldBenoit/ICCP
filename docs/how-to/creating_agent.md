---
layout: default
title: How to create an agent
parent: How-to guides
nav_order: 1
permalink: /docs/how-to/creating_agent
---

# How to create an agent

In this how-to guide, we will give some tips as to how to create your own agent (RL-based or not).


1. Please read the [Agent class](../../../enac-docs/docs/technical-reference/agent/agent_class), [Environment class](../../../enac-docs/docs/technical-reference/environment/environment_class), and [Logger class](../../../enac-docs/docs/technical-reference/logger/logger_class) completely and carefully, otherwise you will not understand the context and meaning of what you're building.

2. If you would like to read some source code of already implemented agents, please read first [Simple Environment](../../../enac-docs/docs/technical-reference/environment/simple_environment) since all agents interact with this environment.

3. If you would like a simple not RL-based agent, please read the source code of `OnOff_Agent.py`.

4. If you would like to implement a RL-based agent with discrete output such as a DQN Agent, plesae read the source of `DQN_Agent.py`.

5. If you would like to implement a RL-based agent with continuous output such as a DDPG Agent, plesae read the source of `DDPG_Agent.py`.

