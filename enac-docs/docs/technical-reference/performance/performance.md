---
layout: default
title: Performance
parent: Technical reference
nav_order: 5
permalink: /docs/technical-reference/performance
---

# Performance

`ICCP` contains a module called `Performance` which contains all functionalities to assess the performance of agents and also easily search through the logs. This functionality of `ICCP` makes working with agents much easier and faster. One can easily iterate to optimize its agents performance and also compare it to other agents.

To understand why and how the performance assessment was defined this way, please read the [background information](../../../enac-docs/docs/background-information/assessment) on `Performance`.

## Class methods and definitions

### Utility functions

To understand why and how utility functions are used, we refer to the [background information](../../../enac-docs/docs/background-information/utility) on it.

`cumulative_reward`
```python
def cumulative_reward(data: pd.DataFrame) -> float:
    """ Given a dataframe containing a reward column, computes cumulative reward"""
```

`negative_cumulative_heating`
```python
def negative_cumulative_heating(data: pd.DataFrame) -> float:
    """ Given a dataframe containing a reward column, computes cumulative reward"""
```


`constant_utility`
```python
def constant_utility(data: pd.DataFrame) -> float:
    """ Constant utility function, to be used when there is no preference between two sessions"""
```

### Statistics computation

`IQR`
```python
def IQR(arr: np.ndarray) -> float:
    """ Computes the inter-quartile range of an array"""
```


`CVaR`
```python
def CVaR(arr: np.ndarray, alpha: float = 0.05) -> float:
    """ Computes the conditional value at risk of an array with risk threshold alpha"""
```

### Across time (during training)

Definitions of methods to evaluate in-training risk and dispersion of a single run.

`compute_dispersion_across_time`
```python
def compute_dispersion_across_time(
    data: pd.DataFrame, column_name: str, window: int
) -> float:
    """We define the measure of dispersion across a training run as the mean of the 
    rolling inter-quartile range.

    Args:
        data (pd.DataFrame): DataFrame containing the specified column
        column_name (str): column name over which dispersion should be computed
        window (int): window length of the rolling inter-quartile range.

    Returns:
        float: the measure of dispersion across time
    """
```


`compute_risk_across_time`
```python
def compute_risk_across_time(
    data: pd.DataFrame, column_name: str, alpha: float, window: int
) -> float:
    """We define the measure of risk across a training run as the mean of the 
    rolling conditional value at risk.

    Args:
        data (pd.DataFrame): DataFrame containing the specified column
        column_name (str): column name over which dispersion should be computed
        window (int): window length of the rolling inter-quartile range.
    Returns:
        float: the measure of risk across time
    """
```


`across_time`
```python
def across_time(
    data: pd.DataFrame,
    window: int = 3 * 6,
    column_name: str = "Reward",
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Given a dataframe summarizing a training session, computes the mean risk and mean dispersion of the specified column
    over a specified time window. 
    
    Why do we use a window? 

    It is such that the computed risk and dispersion still make sense in the context of real-life usage. Indeed, a training session
    may last over 5 months of simulated data but we should assess the behaviour of the controller over the span of a day or hours
    as it is the time span effectively experienced by the human users. In this case, having a window = 3*6 spans 3 hours because the 
    Simple Environment does 6 steps per hour.

    Args:
        data (pd.DataFrame): Summary DataFrame of the training session.
        window (int, optional): Time window over which dispersion and risk are computed. Defaults to 3*6.
        column_name (str, optional): Name of the column over which we compute. Defaults to "Reward".
        alpha (float, optional): Risk threshold. Defaults to 0.05.

    Returns:
        Tuple[float, float]: (dispersion, risk)
    """
```

## Across runs (during training)

Definitions of methods to evaluate in-training risk and dispersion of multiple runs.

`across_runs`
```python
def across_runs(
    agent: Agent,
    agent_config_path: str,  # absolute path to the logging of the agent such that its configuration can be loaded
    parameter: Tuple[str, List[Any]],
    num_episodes: int,
    num_iterations: int = None,
    column_names: List[str] = [
        "Reward",
        "Tset",
    ],  # list of the columns in summary df on which we wish to measure risk and dispersion
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    window: int = 3 * 6,  # window to compute iqr and cvar across time
    alpha=0.05,
) -> Dict[str, Any]:
    """Given an agent initilaized with its environment, the absolute path to the configuration / log we wish to assess,
    loads the correct configuration on the agent. 

    Then, given the parameter name that we wish to vary and the list of the different values it will take, 
    it iterates over all the possible new configurations and:

    - compute the dispersion and risk of the specified column names for each training run / session (by using across_time())
    - compute the dispersion and risk of the utility function over all training runs / sessions

    Args:
        agent (Agent): Agent to be assessed, only needs to be instantiated with the correct environment
        agent_config_path (str):  Absolute path to the configuration / log we wish to assess
        num_episodes (int): number of episodes of each training session
        num_iterations (int, optional): number of iterations of each training session. Defaults to None.
        column_names (List[str], optional): Name of the columns over which we will compute dispersion and risk at each training session.
        Defaults to [ "Reward", "Tset", ].
        utility_function (Callable[[pd.DataFrame], float], optional): Utility function to assess performance of a training session.
        Defaults to cumulative_reward.
        window (int, optional): Time window over which dispersion and risk are computed. Defaults to 3*6.
        alpha (float, optional): Risk threshold. Defaults to 0.05.
    Returns:
        (Dict[str,Any]): dictionary summarizing all results of the performance assessment.
    """
```

## Across fixed policy (after training)

Definitions of methods to evaluate after training risk and dispersion of multiple runs.

`across_fixed_policy`
```python
def across_fixed_policy(
    agent: Agent,
    agent_config_path: str,  # absolute path to the logging of the agent such that its configuration can be loaded
    num_testing: int,
    num_episodes: int,
    num_iterations: int = None,
    column_names: List[str] = [
        "Reward",
        "Tset",
    ],  # list of the columns in summary df on which we wish to measure risk and dispersion
    utility_function: Callable[[pd.DataFrame], float] = cumulative_reward,
    window: int = 3 * 6,  # window to compute iqr and cvar across time
    alpha=0.05,
) -> Dict[str,Any]:
    """Given an agent initilaized with its environment, the absolute path to the configuration / log we wish to assess,
    loads the correct configuration on the agent. 

    It will test the fixed policy num_testing times and:

        - compute the dispersion and risk of the specified column names for each testing run / session (by using across_time())
        - compute the dispersion and risk of the utility function over all testing runs / sessions

    Args:
        agent (Agent): Agent to be assessed, only needs to be instantiated with the correct environment
        agent_config_path (str):  Absolute path to the configuration / log we wish to assess
        num_testing(int): The number of times the fixed policy will be tested
        num_episodes (int): number of episodes of each training session
        num_iterations (int, optional): number of iterations of each testing session. Defaults to None.
        column_names (List[str], optional): Name of the columns over which we will compute dispersion and risk at each testing session.
        Defaults to [ "Reward", "Tset", ].
        utility_function (Callable[[pd.DataFrame], float], optional): Utility function to assess performance of a training session.
        Defaults to cumulative_reward.
        window (int, optional): Time window over which dispersion and risk are computed. Defaults to 3*6.
        alpha (float, optional): Risk threshold. Defaults to 0.05.

    Returns:
        (Dict[str,Any]): dictionary summarizing all results of the performance assessment.
    """
```

### Useful methods

1. How to use `search_paths` is explained in [Searching through logs](../../../enac-docs/docs/how-to/search).

<br>


2. How to use `all_combinations_list` and `search_similar` is explained in [Hyperparameter tuning](../../../enac-docs/docs/how-to/hyper_tuning).

<br>

3. How to use `load_trained_agent`, `load_json_params`, and `load_summary_df` is explained in [Loading from logs](../../../enac-docs/docs/how-to/load).











