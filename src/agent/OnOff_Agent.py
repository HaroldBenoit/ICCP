from calendar import c
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


import os
import pandas as pd

from logger.SimpleLogger import SimpleLogger

from agent.Agent import Agent
from environment.Environment import Environment


class OnOffAgent(Agent):
    def __init__(
        self, env: Environment, is_step: bool = True,
    ):
        """"Simple OnOff Agent. Made to interact with ContinuousEnvironment. Inherits from Agent class.

        The OnOff Agent is a very simple agent.

        It has two possible policies:

        If is_step = True, then the agent will set the temperature to the min temperature when there's no occupancy
        and to the max temperature when there's occupancy. This policy may be called a 'step function'.

        If is_step=False, then the agent will set the temperature as a linear function of occupancy i.e.
        temperature = min_temperature + occupancy*(max_temperature - min_temperature)

        Args:
            env (Environment): Environment inheriting from the Environment class.
            is_step (bool, optional): Whether the policy is a step function or linear function. Defaults to True.
        """

        self.env = env
        self.is_step = is_step

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """We refer to the Agent class docstring."""

        d = self.env.observation_to_dict(state)
        occ = d["Occ"][0]
        if self.is_step:
            # step function
            selected_action = self.env.min_temp if occ == 0.0 else self.env.max_temp
        else:
            # linear function
            selected_action = self.env.min_temp + occ * (
                self.env.max_temp - self.env.min_temp
            )

        return selected_action

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """We refer to the Agent class docstring."""
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def reset(self) -> Agent:
        return self

    def train(
        self,
        logging_path: str,
        num_iterations=None,
        num_episodes=1,
        log=True,
        is_test=False,
    ) -> Tuple[str, pd.DataFrame]:
        """We refer to the Agent class docstring."""
        self.is_test = is_test

        ## check num_iterations
        if num_iterations is None:
            num_iterations = self.env.numsteps

        if num_iterations > self.env.numsteps:
            print(
                f"WARNING: Number of iterations chosen ({num_iterations}) is higher than the number of steps of the environment ({self.env.numsteps}) "
            )
            num_iterations = self.env.numsteps

        ## instantiate logger
        if log: 
            logger = SimpleLogger(
                logging_path=logging_path,
                agent_name="OnOff_Agent",
                num_episodes=num_episodes,
                num_iterations=num_iterations,
            )

        self.opts = {
            "Tair": {"secondary_y": None, "range": [10, 24], "unit": "(°C)",},
            "Tset": {
                "secondary_y": "moving_average",
                "range": [14, 22],
                "unit": "(°C)",
            },
            "PMV": {"secondary_y": None, "range": [-3, 3], "unit": "(-)",},
            "Heating": {"secondary_y": "cumulative", "range": None, "unit": "(kJ)",},
            "Reward": {"secondary_y": "cumulative", "range": [-5, 5], "unit": "(-)",},
            "Occ": {"secondary_y": None, "range": None, "unit": "(-)",},
        }

        summary_df: pd.DataFrame = pd.DataFrame()

        tair = []
        actions = []
        pmv = []
        qheat = []
        rewards = []
        occ = []

        for episode_num in range(num_episodes):

            state = self.env.reset()
            # need to chdir back to logging_path at each episode because calling env.reset() calls chdir() too
            if log:
                os.chdir(logging_path)

            for i in range(num_iterations):

                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                ## keeping track of the value we've seen
                rewards.append(reward)
                actions.append(action)
                pmv.append(info["pmv"][0])
                d = self.env.observation_to_dict(next_state)
                tair.append(d["Tair"][0])
                heat = d["Qheat"][0]
                qheat.append(heat)
                occ.append(d["Occ"][0])

                state = next_state

            ## slicing lower and upper bound
            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations

            summary_df = pd.DataFrame(
                {
                    "Tair": tair[lower:upper],
                    "Tset": actions[lower:upper],
                    "PMV": pmv[lower:upper],
                    "Heating": qheat[lower:upper],
                    "Reward": rewards[lower:upper],
                    "Occ": occ[lower:upper],
                }
            )

            summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

            if log:
                logger.plot_and_logging(
                    summary_df=summary_df,
                    agent=self,
                    episode_num=episode_num,
                    is_summary=False,
                    opts=self.opts,
                )

        # plot a summary that contatenates all episodes together for a complete overview of the training

        summary_df = pd.DataFrame(
            {
                "Tair": tair,
                "Tset": actions,
                "PMV": pmv,
                "Heating": qheat,
                "Reward": rewards,
                "Occ": occ,
            }
        )

        summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

        if log and num_episodes > 1:
            logger.plot_and_logging(
                summary_df=summary_df,
                agent=self,
                episode_num=num_episodes,
                is_summary=True,
                opts=self.opts,
            )

        # self.env.close()

        results_path = logger.RESULT_PATH if log else ""

        return (results_path, summary_df)

    def test(
        self, logging_path: str, num_iterations=None, num_episodes=1, log=True
    ) -> Tuple[str, pd.DataFrame]:
        """We refer to the Agent class docstring."""

        return self.train(
            is_test=True,
            logging_path=logging_path,
            num_iterations=num_iterations,
            num_episodes=num_episodes,
            log=log,
        )

    def seed_agent(self, seed):
        pass

    def save(self, filename, directory):
        pass

    def load(self, filename, directory) -> Agent:
        return self

    def log_dict(self) -> Dict[str, Any]:
        return {}
