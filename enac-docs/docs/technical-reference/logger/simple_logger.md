---
layout: default
title: Simple Logger
parent: Logger
grand_parent: Technical reference
nav_order: 2
permalink: /docs/technical-reference/logger/simple_logger
---

# Simple Logger
The Simple Logger is the basic logger included in `ICCP`. This logger works in conjunction with the [Simple Environment](../../../../enac-docs/docs/technical-reference/environment/simple_environment).


## Class methods and definitions


### Initialization

As said in the [Logger class](../../../../enac-docs//docs/technical-reference/logger/logger_class) description, a logger is instantiated for a single training/testing session and is responsible for logging all the important data from the session.

When initializing the Simple Logger, given this signature : 
```python
def __init__(self,
 logging_path: str,
 agent_name: str, 
 num_episodes: int,
 num_iterations: int):
```
It will create the following folders inside the of the logging path:

┣📂{agent_name}
┃┣ 📂{current_time}
┃ ┃ ┣ 📂results_{current_time}
┃ ┃ ┃ ┣ 📂experiments_csv *(where all the observations and actions are saved)*
┃ ┃ ┃ ┣ 📂model_weights *(neural networks weights)*
┃ ┃ ┃ ┗ 📂plots
┃ ┃ ┃ ┃ ┣ 📂pmv_categories *(plots of the time spent by the agent in different pmv categories)*
┃ ┃ ┃ ┃ ┗ 📂summary *(summarizing plots of all the observations and actions throughout time)*
 

 ### `plot_and_logging`

 This function is usually used at the end of `train()` or `test`and called by the agent itself. 
 
 ```python
     def plot_and_logging(
        self,
        summary_df: pd.DataFrame,
        agent: Agent,
        episode_num: int,
        is_summary: bool,
        opts: Dict[str, Dict[str, Any]],
    ):
        """Main method of the logger that takes care of saving/logging all the data and plotting
        relevant summaries of the data.

        Args:
            summary_df (pd.DataFrame): Should contain all the relevant data needed to log and plot.
            agent (Agent): The agent that was trained/tested/assessed
            episode_num (int): The current episode number of the session (the logger can be called at the end of every episode )
            is_summary (bool): Boolean flag to make sure the titles make sense.
            opts (Dict[str, Dict[str, Any]]): Dictionary defining the plotting parameters.

        """
 ```

 Nonetheless, we will explain how to use it:

 1. Decide what will the `summary_df` DataFrame contain. It will probably contain some parameters of [Simple Environment](../../../../enac-docs/docs/technical-reference/environment/simple_environment) such as "Tair" or "PMV", and the action taken at each step "Tset". Lastly, you may wish to keep track of agent's specific parameters such as "Loss".

 2. `is_summary` should be equal to True, if you're calling this function at the end of your last training / testing episode and would like plots that summarize the entire training / testing session.

 3. `opts` is the most important parameter. It defines what columns of `summary_df` will be plotted and further options for each column to obtain a neat plot.

 Here's an example of an `opts` dictionary:

 ```python
opts = {
            "Tair": {"secondary_y": None,
             "range": [10, 24],
              "unit": "(°C)",
                    },

            "Tset": {
                "secondary_y": "moving_average",
                "range": [14, 22],
                "unit": "(°C)",
                    }
        }
 ```

 The plotting options have the following format:

 ```python
"{column_name}" : {
    "secondary_y": None or "cumulative" or "moving_average"
    "range": None or [low,high]
    "unit": any string describing the unit .e.g "(kJ)"
    }
 ```

For `column_name`:

The column name should be identical to the one used in `summary_df`.

For `"secondary_y"`:

This defines whether there is an auxiliary plot to be drawn on the same plot as the column.

- None -> No auxiliary plot
- "cumulative" -> auxiliary plot is the cumulative value over time (e.g. cumulative heating)
- "moving_average" -> auxiliary plot is a moving average over a 24 timesteps window


For `"range"`:

This defines the range of values on the y-axis that will be plotted.

- None -> no specified range, the range will be from the minimum to the maximum value
- [low,high] -> The range will go from low to high on the y-axis














