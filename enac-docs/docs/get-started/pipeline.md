---
layout: default
title: 3. Full assessment of a controller
parent: Get Started
nav_order: 3
permalink: /docs/get-started/pipeline
---

# Full assessment of a controller

## Introduction

This tutorial serves the purpose of showcasing the entire *research process* from training to performance assessment.

In this tutorial:

1. We import the relevant packages.
2. We instantiate an environment with all its options.
3. We train and do hyperparameter tuning of a controller.
4. Find the best agent.
5. Measure in-training performance of the agent.
6. Measure after-training performance of the agent. 


This tutorial is done using the [`simple_simulation`](../../../enac-docs/docs/technical-reference/environment/simple_environment) EnergyPlus environment as it is readily available in `ICCP`. 

### Imports

```python
import numpy as np
from agent.DQN_Agent import DQNAgent
from environment.DiscreteEnvironment import DiscreteSimpleEnvironment
import pandas as pd
import Performance
```

## Initializing environment

```python
env = DiscreteSimpleEnvironment(param_list=['Tair', 'RH', 'Tmrt', 'Tout', 'Qheat', 'Occ'],
alpha=1,
beta=1,
min_temp=16,
max_temp=21,
discrete_action_dim=100,
modelname='CELLS_v1.fmu',
simulation_path=r'C:\ICCP\EnergyPlus_simulations\simple_simulation',
days=151,
hours=24,
minutes=60,
seconds=60,
ep_timestep=6)
```

## Hyperparameter tuning and training

```python
from Performance import all_combinations_list
from Performance import search_similar

agent_arguments = {
"memory_size": [1000],
"batch_size": [32],
"actor_update":[8],
"target_update": [100],
"epsilon_decay": [1 / 20000],
"max_epsilon": [1],
"min_epsilon":  [0.0],
"lr":[1e-3,5e-3],
"gamma": [0.99],
"inside_dim": [128,256,512],
"num_hidden_layers": [1,2,3,4,5,6,7,8],
"seed": [800]
}

logging_path = r"C:\ICCP\logs\simple_simulation"
searching_directory = r"C:\ICCP\logs\simple_simulation\DQN_Agent\results"

for curr_agent_arguments in all_combinations_list(agent_arguments):

    ## creating the dictionary of parameters against which to check
    agent = DQNAgent(env).from_dict(dict_arguments=curr_agent_arguments)
    log_dict = {**agent.log_dict(), **env.log_dict()}
    num_episodes = 10
    log_dict["num_episodes"] = num_episodes

    ## so that we don't train a configuration that has already been trained
    if(not(search_similar(searching_directory, log_dict))):
        results_path, summary_df = agent.train(
         logging_path= logging_path,
         num_iterations= None,
         num_episodes=num_episodes,
         log=True)
```

## Finding the best performing agent in the logs

We rank agents according to the defined utility function which is the cumulative reward.

```python
searching_directory = r"C:\ICCP\logs\simple_simulation\DQN_Agent"

conditions = {
"seed":["=",800],
"alpha":["=",1],
"beta":["=",1], 
"num_episodes":["=",10]}

best_path_list = Performance.search_paths(
searching_directory,
conditions=conditions,
top_k=1,
utility_function=Performance.cumulative_reward,
normalized=True)
```

## Testing in-training performance pipeline

Here, we test the best agent found above.

```python
from logger.SimpleLogger import SimpleLogger

best_agent_path = best_path_list[0]

agent= Performance.load_trained_agent(
DQNAgent(env),
results_path=best_agent_path)

parameter = ("seed", [775,776,777,778])

logging_path = r"C:\ICCP\logs\simple_simulation"

utility_function = Performance.cumulative_reward
agent = DQNAgent(env=env)
num_episodes = 2
num_iterations = env.numsteps
agent_name = "DQN_Agent"

results_dict = Performance.across_runs(
agent=agent,
agent_config_path=best_agent_path,
parameter=parameter,
num_episodes=num_episodes,
num_iterations=num_iterations,
utility_function=utility_function,
alpha=0.05,
column_names=["Tset,Reward"])

logger = SimpleLogger(
        logging_path=logging_path,
        agent_name="DQN_Agent",
        num_episodes=num_episodes,
        num_iterations=num_iterations,
    )

logger.log_performance_pipeline(
results_dict,
fixed_policy=False)
```

## Testing fixed policy performance

Here, we test the best agent found above.

```python
from logger.SimpleLogger import SimpleLogger

best_agent_path = best_path_list[0]

agent= Performance.load_trained_agent(
DQNAgent(env),
results_path=best_agent_path)

utility_function = Performance.cumulative_reward
agent = DQNAgent(env=env)
num_testing = 3
num_episodes = 5
num_iterations = env.numsteps
agent_name = "DQN_Agent"

results_dict = Performance.across_fixed_policy(
agent=agent,
agent_config_path=best_agent_path,
num_testing=num_testing,
num_episodes=num_episodes,
num_iterations=num_iterations,
utility_function=utility_function,
alpha=0.05,
column_names=["Tset,Reward"])

logging_path = r"C:\ICCP\logs\simple_simulation"

logger = SimpleLogger(
        logging_path=logging_path,
        agent_name="DQN_Agent",
        num_episodes=num_episodes,
        num_iterations=num_iterations,
    )

logger.log_performance_pipeline(
results_dict,
fixed_policy=True)
```

## Conclusion

After doing all steps, one is able to get a thorough review of the performance, dispersion and risk of its controller in-training and after-training.
