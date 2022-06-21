from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Any, TypeVar
import numpy as np
from gym.spaces import Space


# ObsType = TypeVar("ObsType")
# ActType = TypeVar("ActType")


class Environment(metaclass=ABCMeta):
    """Abstract class defining the basic functionalities that a reinforcement learning environment should have.
    Its functionalities are very similar to those of a OpenAI Gym environment. The choice for not implementing
    directly into an OpenAI Gym environment is further discussed into the background information of the docs.
    """

    ## Here, properties are defined to enforce the presence of certain attributes

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Should return the action space of the environment. In this case, it is a Space from the Gym API.
        Spaces are objects that can accurately describe vector spaces, continuous or discrete. More information
        here: https://www.gymlibrary.ml/content/spaces/"""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Should return the observation space of the environment. In this case, it is a Space from the Gym API.
        Spaces are objects that can accurately describe vector spaces, continuous or discrete. More information
        here: https://www.gymlibrary.ml/content/spaces/"""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """ Should return the dimension of the action space i.e. the number of actions taken"""
        ...

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """ Should return the dimension of the observation space"""
        ...

    @abstractmethod
    def reset(self) -> np.ndarray:
        """ Method to reset the environment to its initial state. Should be called before calling step()

        Returns:
            np.ndarray: initial observation / state of the environment.
        """
        pass

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
        pass

    @abstractmethod
    def log_dict(self) -> Dict[str, Any]:
        """Returns a dictionary of all the important parameters of the agent and the corresponding values (similar to from_dict() above).
        This dictionary will be used to log informations about training and testing sessions, therefore important parameters is defined here by 
        whether they are worthy to log. Keep in mind that anything that is logged will be searchable using the search functions, thus it is always
        better to log too much than too little.

        Returns:
            Dict[str, Any]: key: name of parameters, value: value of parameter in the agent
        """
        pass

