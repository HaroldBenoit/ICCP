---
layout: default
title: Logger class
parent: Logger
grand_parent: Technical reference
nav_order: 1
permalink: /docs/technical-reference/logger/logger_class
---

# Logger class
Abstract class defining the basic functionalities that a reinforcement learning logger should have. A logger is instantiated for a single training/testing session and is responsible for logging all the important data from the session. It may also be instantiated in the context of performance assessment.

## Class methods and definitions


`plot_and_logging` is the main method of the logger. It takes care of every aspect of logging and plotting.
```python
@abstractmethod
def plot_and_logging(
    self, summary_df: pd.DataFrame, agent: Agent, episode_num: int, is_summary: bool
) -> None:
    """Main method of the logger that takes care of saving/logging all the data and plotting
    relevant summaries of the data.
    Args:
        summary_df (pd.DataFrame): Should contain all the relevant data needed to log and plot.
        agent (Agent): The agent that was trained/tested/assessed
        episode_num (int): The current episode number of the session (the logger can be called for every episode )
        is_summary (bool): Boolean flag to make sure the titles make sense
    """
```


`RESULT_PATH` is the absolute path where the logger is writing the logs of the training / testing session.
```python
@property
@abstractmethod
def RESULT_PATH(self) -> str:
    """ Should return the absolute path where the logs for the session can be found."""
```

`PERFORMANCE_PATH` is the absolute path where the logger is writing the logs of the performance assessment session.
```python
@property
@abstractmethod
def PERFORMANCE_PATH(self) -> str:
    """ Should return the absolute path where the logs for the performance assessment can be found."""
```









