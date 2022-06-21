---
layout: default
title: How to load from logs
parent: How-to guides
nav_order: 5
permalink: /docs/how-to/load
---


# How to load data structures from logs

## Introduction

After a training / testing session has been finished,  a [Logger](../../../enac-docs/technical-reference/logger) will take care of plotting and saving relevant data. 

To avoid recomputing and make life easier, the `Performance` module has a few "loading" functions to load useful data from the logs.

This is usually the next after having found absolute paths of logs using `search_paths` like described in [Searching through logs](../../../enac-docs/docs/how-to/search).


## Methods

This method allows the user to load an already trained agent given the path to the log of its training session. This allows to avoid recomputing.

`Performance.load_trained_agent`
```python
def load_trained_agent(agent: Agent, results_path: str) -> Agent:
    """Given an initialized agent with its environment and an absolute path of a training session,
        loads the trained data structures of this session.

    Args:
        agent (Agent): Initialized agent with its environment
        results_path (str): Absolute path of the log session of the training session

    Returns:
        Agent: Agent with trained data structures of the training session
    """
```

This method allows to have a quick access to the dictionary of parameters of the logging session.

`Performance.load_json_params`
```python
def load_json_params(results_path: str) -> Dict[str, Any]:
    """Given an absolute path of a log session, loads the json file of parameters as a dictionary.

    Args:
        results_path (str): Absolute path of a log session

    Returns:
        Dict[str, Any]: Dictionary of the parameters in the json file in results_path
    """
```

This method allows to load the pandas DataFrame summarizing the entire log session.

`Performance.load_summary_df`
```python
def load_summary_df(results_path: str) -> pd.DataFrame:
    """Given an absolute path of a log session, loads the summary DataFrame of the session.

    Args:
        results_path (str): Absolute path of a log session

    Returns:
        pd.DataFrame: Summary DataFrame of the training session
    """
```
