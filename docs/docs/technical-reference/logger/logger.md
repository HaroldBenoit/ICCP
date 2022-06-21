---
layout: default
title: Logger
parent: Technical reference
nav_order: 4
permalink: /docs/technical-reference/logger
has_children: true
---

# Loggers

ECP contains a folder called `logger` where all the logic and code about logging reinforcement learning training / testing / performance assessment may be found.

All implemented environments interact with an `EnergyPlus` simulation but the main abstraction called `Environment` can work with any kind of simulation as long as there is `Python` support.

An example is given below as to how you could import elements from `logger`.

```python
# main abstract class
from logger.Logger import Logger
# concrete implementation of a simple Logger made to interact with SimpleEnvironment
from logger.SimpleLogger import SimpleLogger
```





