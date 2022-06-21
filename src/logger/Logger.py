from abc import ABCMeta, abstractmethod
from re import S
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
import pandas as pd
from agent.Agent import Agent


class Logger(metaclass=ABCMeta):
    """Abstract class defining the basic functionalities that a reinforcement learning logger should have.
    A logger is instantiated for a single training/testing session and is responsible for logging all the
    important data from the session. It may also be instantiated in the context of performance assessment.
    """

    @property
    @abstractmethod
    def RESULT_PATH(self) -> str:
        """ Should return the absolute path where the logs for the session can be found."""
        ...

    @property
    @abstractmethod
    def PERFORMANCE_PATH(self) -> str:
        """ Should return the absolute path where the logs for the performance assessment can be found."""
        ...

    @abstractmethod
    def plot_and_logging(
        self, summary_df: pd.DataFrame, agent: Agent, episode_num: int, is_summary: bool
    ) -> None:
        """Main method of the logger that takes care of saving/logging all the data and plotting
        relevant summaries of the data.

        Args:
            summary_df (pd.DataFrame): Should contain all the relevant data needed to log and plot.
            agent (Agent): The agent that was trained/tested/assessed
            episode_num (int): The current episode number of the session (the logger can be called for every episode )
            is_summary (bool): Boolean flag to make sure the titles make sense
        """

        ...

    def log_performance_pipeline(self, results: Dict[str, Any], fixed_policy:bool):
        """Logging method to be used by the performance assessment pipeline

        Args:
            results (Dict[str, Any]): Dictionary of relevant data to be logged
            fixed_policy(bool): Whether we're logging "across_runs" or "fixed_policy"
        """

        suffix= "fixed_policy" if fixed_policy else "across_runs"
        os.makedirs(f"{self.PERFORMANCE_PATH}\{suffix}", exist_ok=True)

        filename = f"performance_results_{self.time}"

        with open(f"{self.PERFORMANCE_PATH}\{suffix}\{filename}.json", "w") as f:
            f.write(json.dumps(results, indent=True))

    @staticmethod
    def pmv_percentages(pmv: np.ndarray) -> pd.DataFrame:
        """ Utility method: 
            Given an 1-dimensional np.ndarray (officially pmv values but could be any float value),
            returns a dataframe containing the following information:
            For the intervals (-2], [-2,-1.5], [-1.5,-1.0], [-1.0,-0.5], [-0.5,0.0], [0.0, 0.5], [0.5, 1.0], [1),
            what percentage of values in the array fall in the given intervals.

            Args:
            pmv (np.ndarray): The 1-D array containing the pmv values (can be any value actually)
         """

        temp = np.array(pmv)
        intervals = []

        length = 8
        lower = -2
        step = 0.5

        ranges = np.zeros(length)

        for i in range(length):

            if i == 0:
                ranges[i] = (temp < lower).sum()
                interval = f"[-inf,{lower}]"
                intervals.append(interval)

            elif i == 7:
                upper = (i - 1) * step + lower
                ranges[i] = (upper <= temp).sum()
                interval = f"[{upper},inf]"
                intervals.append(interval)

            else:
                lower_1 = lower + (i - 1) * step
                upper_1 = lower + (i) * step
                ranges[i] = ((lower_1 <= temp) & (temp < upper_1)).sum()
                interval = f"[{lower_1},{upper_1}]"
                intervals.append(interval)

        ranges = ranges / ranges.sum()

        # assign data
        data = pd.DataFrame({"intervals": intervals, "ranges": ranges})

        return data

    @staticmethod
    def plot_pmv_percentages(
        pmv: np.ndarray, savepath: str, filename: str, plot_title: str
    ):
        """Utility method: 
            Given an 1-dimensional np.ndarray (officially pmv values but could be any float value),
            plots what percentage of values in the array fall in the given intervals.
            Intervals: (-2], [-2,-1.5], [-1.5,-1.0], [-1.0,-0.5], [-0.5,0.0], [0.0, 0.5], [0.5, 1.0], [1)
            

        Args:
            pmv (np.ndarray): 1-D array containing the pmv values (can be any value actually).
            savepath (str): Absolute path where the plot should be saved.
            filename (str): Name of the plot file to be saved. 
            plot_title (str): Title of the plot
        """

        data = Logger.pmv_percentages(pmv)
        # compute percentage of each format
        percentage = []
        for i in range(data.shape[0]):
            pct = data.ranges[i] * 100
            percentage.append(round(pct, 2))

        data["Percentage"] = percentage

        _ = plt.subplots(1, 1, figsize=(15, 7))
        colors_list = [
            "darkred",
            "coral",
            "coral",
            "seagreen",
            "lime",
            "seagreen",
            "coral",
            "darkred",
        ]

        graph = plt.bar(x=data.intervals, height=data.ranges, color=colors_list)

        plt.xlabel("PMV value interval")
        plt.ylabel("Percentage of hours in interval")
        plt.title(plot_title)

        i = 0
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(
                x + width / 2,
                y + height * 1.01,
                str(data.Percentage[i]) + "%",
                ha="center",
                weight="bold",
            )
            i += 1

        plt.savefig(f"{savepath}/{filename}.png", dpi=400)

