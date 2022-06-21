---
layout: default
title: Environment class
parent: Environment
grand_parent: Technical reference
nav_order: 1
permalink: /docs/technical-reference/environment/environment_class
---

# Environment class

The Environment class defines the basic functionalities that a reinforcement learning environment should have. Its functionalities are very similar to those of a OpenAI Gym environment. The choice for not implementing directly into an OpenAI Gym environment is further discussed in [Background information](../../../../ICCP/docs/background-information/openai).


The class implements the classic "agent-environment" loop.

![Agent-environment loop](/assets/rl_diagram_transparent_bg.png "Agent-environment loop (credits to Gym)")

## Class methods and definitions

`action_space` represents the mathematical description of the dimensions and range of values that define our actions in the environment. 
```python
@property
@abstractmethod
def action_space(self) -> Space:
    """Should return the action space of the environment. In this case, it is a Space from the Gym API.
    Spaces are objects that can accurately describe vector spaces, continuous or discrete. More information
    here: https://www.gymlibrary.ml/content/spaces/"""
```

`observation_space` represents the mathematical description of the dimensions and range of values that define our observations of the environment. 
```python
@property
@abstractmethod
def observation_space(self) -> Space:
    """Should return the observation space of the environment. In this case, it is a Space from the Gym API.
    Spaces are objects that can accurately describe vector spaces, continuous or discrete. More information
    here: https://www.gymlibrary.ml/content/spaces/"""
```

`action_dim` is the mathematical dimension of the action space i.e. if an action consists of setting the temperature and humidity of a room, then the action dimension is 2.
```python
@property
@abstractmethod
def action_dim(self) -> int:
    """ Should return the dimension of the action space i.e. the number of actions taken"""
```


`observation_dim` is the mathematical dimension of the observation space i.e. if an observation consists of observing the temperature, humidity, and brightness of a room, then the observation dimension is 3.
```python
@property
@abstractmethod
def observation_dim(self) -> int:
    """ Should return the dimension of the observation space"""
```


`reset` resets the environment to its inital state.
```python
@abstractmethod
def reset(self) -> np.ndarray:
    """ Method to reset the environment to its initial state. Should be called before calling step()
    Returns:
        np.ndarray: initial observation / state of the environment.
    """
```


`step` applies a given action on the environment.
```python
@abstractmethod
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
    """ Given an action contained in the action space, apply it on the environment.
    
    Args:
        action (np.ndarray): Action to be applied on the environment
    Returns:
        Tuple[np.ndarray, float, bool, dict]: 
        - np.ndarray = new observation / state of the environment after applying the action
        - float = reward i.e. the amount of reward returned after taking this action
        - bool = done i.e. whether we have reached the end of the simulation / environment
        - dict = info i.e. any useful additional info 
    """
```


`log_dict` creates the dictionary of importants parameters that need to be logged
```python
@abstractmethod
def log_dict(self) -> Dict[str, Any]:
    """Returns a dictionary of all the important parameters of the agent and the corresponding values (similar to from_dict() above).
    This dictionary will be used to log informations about training and testing sessions, therefore important parameters is defined here by 
    whether they are worthy to log. Keep in mind that anything that is logged will be searchable using the search functions, thus it is always
    better to log too much than too little.
    Returns:
        Dict[str, Any]: key: name of parameters, value: value of parameter in the agent
    """
```