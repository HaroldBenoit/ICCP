---
layout: default
title: Simple Environment
parent: Environment
grand_parent: Technical reference
nav_order: 2
permalink: /docs/technical-reference/environment/simple_environment
---

# Simple Environment

The Simple Environment is the basic functional EnergyPlus environment included in `ICCP`. Most examples work with this environment.

This environment is a wrapper around the EnergyPlus simulation that can be found at `EnergyPlus_simulations/simple_simulation`.

It is described a simple environment because there's only a single HVAC setpoint that's being acted on.

## Description

The case study building is a 1-story building prototype called “Controlled Environments for Living Lab Studies” (CELLS) actually located at the Smart Living Lab site in Fribourg. The building prototype has 2 rooms that are almost identical in size.

### Observation space
The observation space is of dimension 6, it has:

- "Tair" - Air Temperature
- "Rh" - Relative Humidity
- "Tmrt" - Mean Radiant Temperature
- "Tout" - External Ambient Temperature
- "Qheat" - Heating Demand from the HVAC system
- "Occ" - Occupancy

### Action space

The action space is of dimension 1, it has:

- "Tset", sets the temperature of a HVAC unit located in one of the rooms.


### Thermal comfort or Predicted mean vote (PMV)

One may define the thermal comfort of a room using [Berkeley's PMV method](https://comfort.cbe.berkeley.edu/) which defines the thermal comfort using parameters such as mean radiant temperature or relative humidity.

The PMV usually ranges from -2 to 2, where 0 is neutral, -2  very cold, and 2 is very hot.

In terms of thermal comfort and human health, a room in the PMV range of -0.5 to 0 is considered the best.

### Reward function


The reward of a given observation / state is a weighted combination of the thermal comfort (defined as the PMV) and the heating demand.

Alpha is the parameter for thermal comfort and Beta is the parameter for the heating demand.

Then, we may define the reward as:

Reward = Beta * (1 - (heating/(800'000))) + Alpha * (1 - abs((pmv + 0.5))) * occupancy


Thus, a high Beta heavily penalizes heating and a high Alpha heavily prioritizes thermal comfort.


