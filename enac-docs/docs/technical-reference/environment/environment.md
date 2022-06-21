---
layout: default
title: Environment
parent: Technical reference
nav_order: 3
permalink: /docs/technical-reference/environment
has_children: true
---

# Environments

`ICCP` contains a folder called `environment` where all the logic and code about reinforcement learning environments may be found.

All implemented environments interact with an `EnergyPlus` simulation but the main abstraction called `Environment` can work with any kind of simulation as long as there is `Python` support.

An example is given below as to how you could import elements from `environment`.

```python
# main abstract class
from environment.Environment import Environment
# concrete implementation of a simple EnergyPlus environment
from environment.DiscreteEnvironment import DiscreteSimpleEnvironment
```





