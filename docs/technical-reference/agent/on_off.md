---
layout: default
title: On-Off Agent
grand_parent: Technical reference
parent: Agent
nav_order: 4
permalink: /docs/technical-reference/agent/on_off
---

# On-Off Agent

The On-Off Agent,`OnOff_Agent.py`, implements all the functionalities of the [Agent class](../../../../ICCP/docs/technical-reference/agent/agent_class). 

It is a non-RL-based controller, it has two possible behaviours:

- "step = True" : It sets the temperature to the minimum when the occupancy is zero, otherwise it's the max temperature.

- "step = False": It sets the temperature as a linear function of occupancy, with the minimum when there's no occupancy and linearly increasing until the maximum temperature when there's max occupancy.


The current implementation is compatible with the [Simple Environment](../../../../ICCP/docs/technical-reference/environment/simple_environment), more precisely the `ContinuousSimpleEnvironment`.






