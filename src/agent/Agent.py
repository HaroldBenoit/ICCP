from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class Agent(metaclass=ABCMeta):
    """Abstract class defining the basic functionalities that a reinforcement learning agent should have
    """

    @abstractmethod
    def select_action(self, state: np.ndarray):
        """Given a state/observation of the environment, return the optimal action according to the policy

        Args:
            state (np.ndarray): State/observation of the environment
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        """Given an action, applies the action on the environment to update it.

        Args:
            action (np.ndarray): Action that can be applied on the environment (e.g. new temperature of the room)
        """
        pass

    @abstractmethod
    def reset(self) -> Agent:
        """ Reset the agent to its initial state (where it hasn't been trained and such)"""

    def from_dict(self, dict_arguments: Dict[str, Any]) -> Agent:
        """Given a dictionary of key value pairs where the key is the name of an agent attribute
         (e.g. {"memory_size": 1000, "batch_size": 32,}), return a new agent with the attributes are updated with the new values.

        Args:
            dict_arguments (Dict[str, Any]): key: name of attribute, value: new value

        Returns:
            Agent: A new agent updated with the new attributes.
        """

        for k, v in dict_arguments.items():
            setattr(self, k, v)

        return self.reset()

    @abstractmethod
    def train(
        self, logging_path: str, num_episodes: int, num_iterations: int, log: bool
    ) -> Tuple[str, pd.DataFrame]:
        """Method to train the agent and potentially log its progress.

        Args:
            logging_path (str): Absolute path where the logs should be created (e.g. ./DIET_Controller/logs/simple_simulation).
            num_episodes (int): Number of episodes the agent will be trained.
            num_iterations (int): Number of iterations per episode (By convention, if num_iterations = None, default to the maximum number of iterations).
            log (bool): If true, the entire training session will be logged.

        Returns:
            Tuple[str, pd.DataFrame]: 
            1. The aboslute path of where the logs of the training session may be found.
            2. A pandas DataFrame containing a summary of the important parameters of the session (to be defined depending on the agent and the use case)
        """
        pass

    @abstractmethod
    def test(
        self, logging_path: str, num_episodes: int, num_iterations: int, log: bool
    ) -> Tuple[str, pd.DataFrame]:
        """Method to test the agent and potentially log its results.

        Args:
            logging_path (str): Absolute path where the logs should be created (e.g. ./DIET_Controller/logs/simple_simulation).
            num_episodes (int): Number of episodes the agent will be tested.
            num_iterations (int): Number of iterations per episode (By convention, if num_iterations = None, default to the maximum number of iterations)
            log (bool): If true, the entire testing session will be logged

        Returns:
            Tuple[str, pd.DataFrame]: 
            1. The aboslute path of where the logs of the testing session may be found.
            2. A pandas DataFrame containing a summary of the important parameters of the session (to be defined depending on the agent and the use case)
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

    @abstractmethod
    def seed_agent(self, seed: int):
        """Method to seed all the possible RNG components of the agent. Examples of RNG components to seed would be:
        1. Numpy 2. Pytorch 3. Random 4. Gym spaces

        Args:
            seed (int):
        """
        pass

    @abstractmethod
    def save(self, directory: str, filename: str):
        """ Given a directory and a filename, saves all revelant data structures (e.g. neural networks). 

        Args:
            directory (str): Absolute path where the data structures should be saved (e.g. /logs/2022_06_06/)
            filename (str): Prefix for all the filenames of the saved data structures (e.g. "episode_1")
        """
        pass

    @abstractmethod
    def load(self, directory: str, filename: str):
        """ Given a directory and a filename, load all revelant data structures (e.g. neural networks). 

        Args:
            directory (str): Absolute path where the data structures are to be loaded (e.g. /logs/2022_06_06/)
            filename (str): Prefix for all the filenames of the saved data structures (e.g. "episode_1")
        """
        pass

    # Simple python properties to ensure easy attribute access

    def __getattribute__(self, attr):
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        object.__setattr__(self, attr, value)

