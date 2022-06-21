---
layout: default
title: How to tune hyperparameters 
parent: How-to guides
nav_order: 6
permalink: /docs/how-to/hyper_tuning
---


# How to tune hyperparameters

#### Introduction

After defining a reinforcement learning agent, its environment, and making sure it's functional, the natural next step is __hyperparameter tuning__.

__Hyperparameter tuning__ is a fancy way to talk about finding the best parameters of your agent that maximize your utility function.

This process is fairly computationally intensive and requires testing a large number of parameters, usually by a procedure called [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search).

This is where `all_combinations_list` and `search_similar` comes to aid in this process and automate the grid search process.


## Definitions

Here is the definition of the `all_combinations_list` function.

```python

def all_combinations_list(arguments: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Given a dictionary of type Dict[str, List[Any]], 
    outputs a list of all the combinatorial combinations
    (cartesian product) of the elements in the list.
    This is useful in the case of trying many different 
    combinations of parameter for a reinforcement learning agent.

    Example:
    Given arguments: {"a":[1,2], "b":[3,4]}

    Outputs: [{"a":1, "b":3}, {"a":1, "b":4}, {"a":2, "b":3}, {"a":2, "b":4}]
    

    Args:
        arguments (Dict[str, List[Any]]): Dictionary containing key-value pairs where the key is a string 
        and the value is a list of parameters.

    Returns:
        List[Dict[str,Any]]: Cartesian product of the elements.
    """
```

Here is the definition of `search_similar` function.

```python
def search_similar(searching_directory: str, subset_log_dict: Dict[str, Any]) -> bool:
    """Utilities function to be used when hyperparameter tuning to avoid training already trained agent specifications.

    Given a searching directory and dictionary of parameters, will search the directory to see
    if any logged parameter dictionary contains the specified parameters. 
    
    For example, one can specify subset_log_dict as the parameters of an agent to be trained and
    the searching directory the directory of logs of this agent. Then, one may use this function
    to avoid training already trained agent specifications by first checking if it already exists
    in the logs.

    Args:
        searching_directory (str): Absolute path to the directory where the logs should be searched
        subset_log_dict (Dict[str, Any]): Dictionary of parameters.

    Returns:
        bool: If True, there exists a log containing those parameters.
    """
```

## Usage

The main idea is to leverage the `from_dict` method from the [Agent class](../../../ICCP/docs/technical-reference/agent/agent_classs) to easily give the dictionary of arguments (that we obtain from `all_combinations_list`) to an Agent. 

`search_similar` allows us to avoid recomputation of already trained agents.

An example is given below:

```python
from Performance import all_combinations_list

agent_arguments = {
"inside_dim": [128,256,512],
"num_hidden_layers": [1,2,3],
"seed": [778]
}

logging_path = r"C:\ICCP\logs\simple_simulation"
searching_directory = r"C:\ICCP\logs\simple_simulation\DQN_Agent\results"

for curr_agent_arguments in all_combinations_list(agent_arguments):

    agent = DQNAgent(env).from_dict(dict_arguments=curr_agent_arguments)

    ## creating the dictionary of parameters against which to check
    log_dict = {**agent.log_dict(), **env.log_dict()}
    num_episodes = 10
    log_dict["num_episodes"] = num_episodes

        ## checking if logs already exist
        if(not(search_similar(searching_directory, log_dict))):
            results_path, summary_df = agent.train(logging_path= logging_path,
             num_episodes=num_episodes,
              num_iterations=None,
              log=True)
```







