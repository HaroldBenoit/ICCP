from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
from typing import Any, Dict, List, Tuple

from environment.SimpleEnvironment import SimpleEnvironment


class DiscreteSimpleEnvironment(SimpleEnvironment):
    """ Version of the SimpleEnvironment where the action space is discrete.
    This environment is to be used with the DQN agent, for example."""

    def __init__(
        self,
        param_list: List[str] = [
            "Tair",
            "RH",
            "Tmrt",
            "Tout",
            "Qheat",
            "Occ",
        ],  # what we get from the model at each step
        discrete_action_dim: int = 200,  # defines the granularity of our action space
        min_temp: int = 16,  # minimum temperature for action
        max_temp: int = 21,
        alpha: float = 1,  # thermal comfort
        beta: float = 1,  # energy consumption
        modelname: str = "CELLS_v1.fmu",
        # where the EnergyPlus FMU can be found
        simulation_path: str = r"C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller\EnergyPlus_simulations\simple_simulation",
        # parameters to be found in the idf file
        days: int = 151,
        hours: int = 24,
        minutes: int = 60,
        seconds: int = 60,
        ep_timestep: int = 6,
    ):
        # initialzing the super class
        super().__init__(
            param_list,
            min_temp,
            max_temp,
            alpha,
            beta,
            modelname,
            simulation_path,
            days,
            hours,
            minutes,
            seconds,
            ep_timestep,
        )

        # defining the discrete action space
        self._discrete_action_dim = discrete_action_dim
        self._action_space = Discrete(self.discrete_action_dim)

        # translation from discrete coordinates to temperature
        # i.e. [0,100] -> [16,21]
        self.action_to_temp = np.linspace(
            self.min_temp, self.max_temp, self.discrete_action_dim
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def discrete_action_dim(self):
        return self._discrete_action_dim

    @discrete_action_dim.setter
    def action(self, dim):
        """ Setter for discrete_action. It is a special function because other components need to be updated 
        if the action dim is modified
        """
        self._action_dim = dim
        self._action_space = Discrete(self.discrete_action_dim)
        self.action_to_temp = np.linspace(
            self.min_temp, self.max_temp, self.discrete_action_dim
        )

    @property
    def min_temp(self):
        return self._min_temp

    @min_temp.setter
    def min_temp(self, temp):
        """ Setter for minimum temperature. It is a special function because other components need to be updated 
        if the minimum temperature is modified.
        """
        self._min_temp = temp
        self.action_to_temp = np.linspace(
            self.min_temp, self.max_temp, self.discrete_action_dim
        )

    @property
    def max_temp(self):
        return self._max_temp

    @max_temp.setter
    def max_temp(self, temp):
        """ Setter for maximum temperature. It is a special function because other components need to be updated 
        if the maximum temperature is modified.
        """
        self._max_temp = temp
        self.action_to_temp = np.linspace(
            self.min_temp, self.max_temp, self.discrete_action_dim
        )

    def step(self, action: Discrete) -> Tuple[np.ndarray, float, bool, dict]:
        """ Does the step, converts discrete input into continuous temperature for the simuation"""
        action_temperature = self.action_to_temp[action]
        return super().step(action=action_temperature)

    def log_dict(self):
        """We refer to the Environment class docstring."""

        log_dict = super().log_dict()

        log_dict["discrete_action_dim"] = self.discrete_action_dim
        return log_dict

