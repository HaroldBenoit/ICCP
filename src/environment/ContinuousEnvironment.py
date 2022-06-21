from gym.utils import seeding
from gym.spaces import Discrete, Box
import numpy as np
from typing import Any, Dict, List, Tuple

from environment.SimpleEnvironment import SimpleEnvironment


class ContinuousSimpleEnvironment(SimpleEnvironment):
    """ Version of the SimpleEnvironment where the action space is continuous.
    This environment is to be used with the DDPG agent or the On-Off agent, for example."""

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

        # defining the continuous action space
        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))

    @property
    def action_space(self):
        return self._action_space

    @property
    def min_temp(self):
        return self._min_temp

    @min_temp.setter
    def min_temp(self, temp):
        """ Setter for minimum temperature. It is a special function because other components need to be updated 
        if the minimum temperature is modified.
        """
        self._min_temp = temp
        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))

    @property
    def max_temp(self):
        return self._max_temp

    @max_temp.setter
    def max_temp(self, temp):
        """ Setter for maximum temperature. It is a special function because other components need to be updated 
        if the maximum temperature is modified.
        """
        self._max_temp = temp
        self._action_space = Box(low=self.min_temp, high=self._max_temp, shape=(1,))




