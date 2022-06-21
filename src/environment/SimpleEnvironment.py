from abc import ABCMeta, abstractmethod
import os
from gym.spaces import Discrete, Box
import numpy as np
import math
from typing import Any, Dict, List, Tuple

import math
from pyfmi import load_fmu

from environment.Environment import Environment


class SimpleEnvironment(Environment):
    """ EnergyPlus simple environment. Made to interact with SimpleLogger. Inherits from Environment class.
    This environment is a wrapper around the EnergyPlus simulation that can be found at EnergyPlus_simulations/simple_simulation.
    It is a simple environment because there's only a single HVAC setpoint that's being acted on. """

    def __init__(
        self,
        # list of parameters to be fetched from the EnergyPlus simulation
        # defines the observation space
        param_list: List[str] = [
            "Tair",  # air temperature
            "RH",  # relative humidity
            "Tmrt",  # mean radiant temperature
            "Tout",  # External ambiant temperature
            "Qheat",  # Heating demand from HVAC system
            "Occ",  # Occupancy of the room
        ],  # what we get from the model at each step
        min_temp: int = 16,  # minimum temperature for action
        max_temp: int = 21,  # maximum temperature for action
        alpha: float = 1,  # thermal comfort
        beta: float = 1,  # energy consumption
        modelname: str = "CELLS_v1.fmu", # name of the fmu
        # simulation_path is where the EnergyPlus FMU can be found
        simulation_path: str = r"C:\Users\Harold\Desktop\ENAC-Semester-Project\DIET_Controller\EnergyPlus_simulations\simple_simulation",
        # parameters to be found in the idf file
        days: int = 151,
        hours: int = 24,
        minutes: int = 60,
        seconds: int = 60,
        ep_timestep: int = 6,
    ):

        ## parameters for the EnergyPlus FMU simulation
        self.modelname = modelname
        self.simulation_path = simulation_path

        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.ep_timestep = ep_timestep

        self.numsteps = (
            days * hours * ep_timestep
        )  # total number of simulation steps during the simulation

        self.timestop = (
            days * hours * minutes * seconds
        )  # total time length of our simulation

        self.secondstep = (
            self.timestop / self.numsteps
        )   

        self.simtime = 0  # keeps track of current time in the simulation
        self.model = None

        self._min_temp = min_temp
        self._max_temp = max_temp

        ## parameters for dimensions of the state and reward function
        self.alpha = alpha
        self.beta = beta

        ## defines the data that we fetch from the model at each step to build our observation
        self.param_list = param_list

        ## defining the observation space as a continuous space
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,)
        )

        ## need to reset the environment first, we will keep track of it
        self.has_reset = False
        self.curr_obs = np.array([])

    @property
    def observation_space(self):
        """We refer to the Environment class docstring."""

        return self._observation_space

    @property
    def action_dim(self):
        """We refer to the Environment class docstring."""

        return 1

    @property
    def observation_dim(self):
        """We refer to the Environment class docstring."""

        return len(self.param_list)

    def reset(self, seed=42) -> np.ndarray:
        """
        Resets the environment to an initial state and returns an initial observation.

        Arguments:
        seed(int,optional): seed for seeding the action and observation space

        Returns:
            np.array: Element of self.observation_space, representing the HVAC environment dynamics.
        """

        self.has_reset = True

        ## seeding the spaces
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        ## resetting
        self.simtime = 0  # resetting simulation time tracker

        ## getting to the right place for loading
        os.chdir(self.simulation_path)
        self.model = load_fmu(self.modelname)
        opts = self.model.simulate_options()  # Get the default options
        opts["ncp"] = self.numsteps  # Specifies the number of timesteps
        opts["initialize"] = True
        simtime = 0

        ## critical steps
        self.model.reset()
        self.model.instantiate_slave()
        self.model.initialize(simtime, self.timestop)

        ## getting first observation
        self.curr_obs = np.array(list(self.model.get(self.param_list)))

        return self.curr_obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Run one timestep of the environment’s dynamics. When end of episode is reached, 
        you are responsible for calling reset() to reset this environment’s state. 
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:

        action(object): an action provided by the agent. 

        Return:

        observation (np.ndarray): agent’s observation of the current environment. 
        This will be an element of self.observation_space, representing the HVAC environment dynamics.
        reward (float) : Amount of reward returned as a result of taking the action.
        done (bool): Whether the simulation episode has ended, in which case further step() 
        calls will return undefined results. 
        info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). 
        """

        # checking if environment has been reset
        if not (self.has_reset):
            raise InitializationError("Must reset the environment before using it")

        ## clipping in the correct range
        action = np.clip(action, self._min_temp, self._max_temp)

        if not (isinstance(action, float)):
            action = action.ravel()[0]

        ## setting the HVAC setpoint temperature (name defined in the idf file)
        self.model.set("Thsetpoint_diet", action)
        self.model.do_step(
            current_t=self.simtime, step_size=self.secondstep, new_step=True
        )

        ## getting the current observation from the EnergyPlus environment
        self.curr_obs = np.array(list(self.model.get(self.param_list)))

        ## computing reward after taking action

        pmv = self.comfPMV(self.curr_obs)
        reward = self.compute_reward(self.curr_obs, self.alpha, self.beta)

        self.simtime += self.secondstep

        next_state = self.curr_obs

        ## defines whether it's time to reset the environment or not
        done = self.simtime == self.timestop
        ## debugging dict
        info = {"Simtime": self.simtime, "pmv": pmv}

        return next_state, reward, done, info

    def log_dict(self) -> Dict[str,Any]:
        """We refer to the Environment class docstring."""

        log_dict = {
            "param_list": self.param_list,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "min_temp": self._min_temp,
            "max_temp": self._max_temp,
            "alpha": self.alpha,  # thermal comfort
            "beta": self.beta,  # energy consumption
            "modelname": self.modelname,
            "days": self.days,
            "hours": self.hours,
            "minutes": self.minutes,
            "seconds": self.seconds,
            "ep_timestep": self.ep_timestep,
        }

        return log_dict

    def observation_to_dict(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Given an np.array of the current observation, 
        returns a dictionary with the key being the string description of each element.
        This helps for logging purposes.

        Args:
            obs (np.array): observation of the environment, must be an element of self.observation_space
        """
        dict_values = {self.param_list[i]: obs[i] for i in range(len(self.param_list))}

        return dict_values

    def compute_reward(self, obs: np.ndarray, alpha: float, beta: float):
        """
        Given an observation of the environment, computes the reward
        based on energy consumption and thermal comfort.

        Args:
            obs (Box): observation of the environment
            alpha (float): parameter for thermal comfort
            beta (float): parameter for energy consumption

        Returns:
            reward(float): reward for the given observation
        """
        pmv = self.comfPMV(obs)

        dict_values = self.observation_to_dict(obs)
        qheat_in = dict_values["Qheat"]
        occ_in = dict_values["Occ"]

        reward = (
            beta * (1 - (qheat_in / (800 * 1000)))
            + alpha * (1 - abs((pmv + 0.5))) * occ_in
        )

        return reward

    def comfPMV(self, obs: np.ndarray):
        """ Utility function to compute thermal comfort according to https://comfort.cbe.berkeley.edu/"""

        dict_values = self.observation_to_dict(obs)

        ta = dict_values["Tair"]
        tr = dict_values["Tmrt"]
        vel = 0.1
        rh = dict_values["RH"]
        met = 1.1
        clo = 1
        wme = 0

        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

        icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
        m = met * 58.15  # metabolic rate in W/M2
        w = wme * 58.15  # external work in W/M2
        mw = m - w  # internal heat production in the human body
        if icl <= 0.078:
            fcl = 1 + (1.29 * icl)
        else:
            fcl = 1.05 + (0.645 * icl)

        # heat transfer coefficient by forced convection
        hcf = 12.1 * math.sqrt(vel)
        taa = ta + 273
        tra = tr + 273
        # we have verified that using the equation below or this tcla = taa + (35.5 - ta) / (3.5 * (6.45 * icl + .1))
        # does not affect the PMV value
        tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100.0, 4))
        xn = tcla / 100
        xf = tcla / 50
        eps = 0.00015

        n = 0
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
            if hcf > hcn:
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
            n += 1
            if n > 150:
                print("Max iterations exceeded")
                return 1  # fixme should not return 1 but instead PMV=999 as per ashrae standard

        tcl = 100 * xn - 273

        # heat loss diff. through skin
        hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
        # heat loss by sweating
        if mw > 58.15:
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0
        # latent respiration heat loss
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        # dry respiration heat loss
        hl4 = 0.0014 * m * (34 - ta)
        # heat loss by radiation
        hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
        # heat loss by convection
        hl6 = fcl * hc * (tcl - ta)

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

        return pmv


class InitializationError(Exception):
    """ Simple custom error to enforce reset of the environment before usage"""

    pass
