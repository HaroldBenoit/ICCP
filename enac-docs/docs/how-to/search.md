---
layout: default
title: How to search through logs
parent: How-to guides
nav_order: 4
permalink: /docs/how-to/search
---


# How to search and filter logs

#### Introduction

After a training / testing session has been finished,  a [Logger](../../../enac-docs/docs/technical-reference/logger/logger_class) will take care of plotting and saving relevant data. 

After repeating this process multiple times, one may find itself with a huge number of log folders and no easy way to search through them and rank them.

This is where `search_paths` comes in play. 

After a session, the [Logger](../../../enac-docs/docs/technical-reference/logger/logger_class) will create a json file (e.g. `env_params_2022_5_10_15_0.json`) containing all relevant experiment parameters.

Here's a small snippet of such a json file.

```json
{
"num_episodes": 1,
"num_iterations": 21744,
"is_test": false,
"alpha": 1,
"beta": 2,
and so on ...
}
```

We will leverage this json file to define conditions on different parameters present in this file. For example, we will be able to define the condition "alpha > 1" or "is_test = True" and get back all sessions satisfiying those conditions.

#### Definition

Here is the definition of the `search_paths` function.

```python

def search_paths(
    searching_directory: str,
    conditions: Dict[str, Any],
    utility_function: Callable[[pd.DataFrame], float] = constant_utility,
    top_k: int = -1,
) -> List[str]:
    """ Finds all absolute paths in searching_directory of agent sessions that satisfy the specified conditions.
        Outputs top_k (all if -1) absolute paths ranked according to the utility function
    Args:
        searching_directory (str): Absolute path of the relevant directory where the logs of interest may be found
        conditions (Dict[str,Any]): Conditions that the session must satisfy. Further details on how to define them below.
        utility_function (Callable[[pd.DataFrame], float], optional): Utility function to rank sessions.
        Defaults to constant_utility i.e. no ranking.
        top_k (int, optional): Number of outputted paths. Defaults to 0. If -1, every path is outputted.
        normalized (bool, optional): If True, then the utility function output is divided (i.e. "normalized) by the number of episodes 
        of the session. Defaults to False. If False, does nothing.

    Returns:
        List[str]: All the absolute paths of the sessions logs that satisfy the defined conditions.
    """
```

The key part of the `search_paths` is defining the conditions. These are directly linked with the mentioned json file above. The best way to learn how to define them is by example. An example is given below.

#### Example

```python

import Performance

searching_directory = r"C:\Users\DIET_Controller"

 conditions={
     "alpha": ["<",20], # sessions where alpha was less than 20
     "beta": [">",2], # sessions where beta was bigger than 2
     "num_iterations": ["=",21744], # sessions where the whole simulation episode was used
     "is_test": ["=", True] # only testing sessions 
 }

## This example is specific to SimpleEnvironment
## One may define 
 conditions["pmv"] = {
         "[-inf,-2]": ["<",0.2], # less than 20% of the time spent in the [-inf,-2] interval
         "[-0.5,0.0]": [">", 0.5] # more than 50% of the time spent in the [-0.5,0.0] interval
 }

# Possible intervals are = ['[-inf,-2]', '[-2.0,-1.5]', '[-1.5,-1.0]',
# '[-1.0,-0.5]', '[-0.5,0.0]', '[0.0,0.5]', '[0.5,1.0]', '[1.0,inf]']

## This will return the list of all absolute paths of log folders satisfying the above conditions.
 path_list = Performance.search_paths(searching_directory,conditions)

## This will return the top 5 best absolute paths (according to the normalized utility function) of log folders satisfying the above conditions.
sorted_path_list = Performance.search_paths(searching_directory,conditions, utility_function=Performance.cumulative_reward, top_k = 5, normalized=True)
```

Furthermore, one should be aware when using "cumulative" [utility functions](../../../enac-docs/docs/background-information/utility), there will be a bias towards sessions with a larger number of episodes if `normalized= False`. Putting `normalized = True` corrects this bias.

#### Conclusion

To conclude, the workflow when using `search_paths` is:

- Look at the json files in the logs you want to filter.
- Define conditions based on the parameters present in it.
- (Optional) Define (or use a premade one) a special utility function to rank the sessions and define the wanted number of sessions returned.





