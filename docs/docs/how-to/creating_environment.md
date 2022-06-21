---
layout: default
title: How to create an environment
parent: How-to guides
nav_order: 2
permalink: /docs/how-to/creating_environment
---

# How to create an environment

In this how-to guide, we will give some tips as to how to create your own environment (EnergyPlus or not).


1. Please read the [Agent class](../../../enac-docs/docs/technical-reference/agent/agent_class), [Environment class](../../../enac-docs/docs/technical-reference/environment/environment_class), and [Logger class](../../../enac-docs/docs/technical-reference/logger/logger_class) completely and carefully, otherwise you will not understand the context and meaning of what you're building.

2. If you would like to read some source code of an already implemented environment, please read first [Simple Environment](../../../enac-docs/docs/technical-reference/environment/simple_environment) since it's the basic environment of `ICCP`. Its source code `SimpleEnvironment.py` may be informative for the more involved programming details.

3. If you would like to know how to implement an environment compatible with a discrete ouput agent (e.g. DQN Agent), please read the source code of `DiscreteEnvironment.py`.

4. If you would like to know how to implement an environment compatible with a continuous ouput agent (e.g. DDPG Agent), please read the source code of `ContinuousEnvironment.py`.



## Dealing with EnergyPlus FMUs

Dealing with FMUs is a tricky business as the documentation is obscure or non-existent.  

## Instantating / resetting the EnergyPlus FMU

Countless hours have been lost by the author trying to figure out how to reset a FMU environment without killing the Python kernel but a solution was found. Here's how to do it.

- `simulation_path` = absolute path where the EnergyPlus FMU can be found
- `modelname` = name of the FMU i.e. it must end with ".fmu"
- `timestop` = total time length of the simulation in seconds

```python
    ## getting to the right place for loading
    os.chdir(simulation_path)
    ## 
    model = load_fmu(modelname)
    opts = model.simulate_options()  # Get the default options
    opts["ncp"] = numsteps  # Specifies the total number of timesteps during simulation
    opts["initialize"] = True
    simtime = 0

    ## Critical steps to ensure the environment resets correctly
    self.model.reset()
    self.model.instantiate_slave()
    self.model.initialize(simtime, timestop)
```

## Applying action on the FMU

Here's how to apply an action.

- `simtime` = current simulation time in seconds
- `secondstep` = length of a single step in seconds = (total length of simulation) / (total number of steps during simulation)

```python
## setting the HVAC setpoint temperature ("Thsetpoint_diet is defined in the idf file)
self.model.set("Thsetpoint_diet", action)
self.model.do_step(
    current_t=simtime, step_size=secondstep, new_step=True
)
```

## Getting observation / state from FMU

Here's how to get an observation.

```python
# here we use the list of parameters defined in Simple Environment
observation = list(model.get(["Tair","RH","Tmrt", "Tout","Qheat","Occ"]))
```