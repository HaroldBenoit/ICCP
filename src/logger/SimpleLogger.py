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
from logger.Logger import Logger


class SimpleLogger(Logger):
    """ Simple logger. Made to interact with SimpleEnvironment. Inherits from Logger class."""

    def __init__(
        self, logging_path: str, agent_name: str, num_episodes: int, num_iterations: int
    ):

        self.num_episodes = num_episodes
        self.num_iterations = num_iterations

        ## Getting current time to set up logging directory for the current session
        date = datetime.now()
        temp = list([date.year, date.month, date.day, date.hour, date.minute])
        temp = [str(x) for x in temp]
        self.time = "_".join(temp)

        ## Setting up important absolute path attributes
        self.TIME_PATH = f"{str(date.year)}_{str(date.month)}_{str(date.day)}"
        self._RESULT_PATH = (
            f"{logging_path}/{agent_name}/results/{self.TIME_PATH}/results_{self.time}"
        )
        self._PERFORMANCE_PATH = (
            f"{logging_path}/{agent_name}/performance_results"
        )

        ## Creating directories where the logging will be done
        os.makedirs(self.RESULT_PATH, exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/summary", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/pmv_categories", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/experiments_csv", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/model_weights", exist_ok=True)

    @property
    def RESULT_PATH(self):
        return self._RESULT_PATH

    @property
    def PERFORMANCE_PATH(self):
        return self._PERFORMANCE_PATH

    def plot(
        self,
        summary_df: pd.DataFrame,
        opts: Dict[str, Dict[str, Any]],
        plot_filename: str,
        title: str,
    ):
        """Utility function to plot all the relevant data.
        It plots:
        All the relevant data present in summary_df, with the specifities of the plots defined in opts

        Args:
            summary_df (pd.DataFrame):  A pandas DataFrame containing a summary of the important parameters of the session
            (to be defined depending on the agent and the use case)
            opts (Dict[str, Dict[str, Any]]): A dictionary where the key is the column name of one of the columns in summary_df 
            and the value is the corresponding plotting options for the column.

            The plotting options have the following format:

            "column_name" : {
                "secondary_y": None or "cumulative" or "moving_average"
                "range": None or [low,high]
                "unit": any string describing the unit .e.g "(kJ)"
            }

            For "secondary_y":
            This defines whether there is an auxiliary plot to be drawn on the same plot as the column.
                None -> No auxiliary plot
                "cumulative" -> auxiliary plot is the cumulative value over time (e.g. cumulative heating)
                "moving_average" -> auxiliary plot is a moving average over a 24 timesteps window

            For "range":
            This defines the range of values on the y-axis that will be plotted
                None -> no specified range, the range will be from the minimum to the maximum value
                [low,high] -> The range will go from low to high on the y-axis


            plot_filename (str): plot filename 
            title (str): plot title
        """

        num_rows = len(summary_df.columns)

        # Preparing the canvas
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            # specifiying which part of the plot will contain an auxiliary plot
            specs=[
                [{"secondary_y": (opts[column_name]["secondary_y"] is not None)}]
                for column_name in summary_df.columns
            ],
        )

        ## This list defines a list of color pairs where the first color is for the primary plot
        ## and the second color is for the auxiliary plot. They were specifically chosen to fit
        ## well together.
        ## Please add more if the plot has more 8 columns to be plotted.
        ## You may find more colors here: https://matplotlib.org/stable/gallery/color/named_colors.html
        colors = [
            ("cyan", "dark cyan"),
            ("fuchsia", "purple"),
            ("gold", "darkgoldenrod"),
            ("red", "darkred"),
            ("lime", "darkgreen"),
            ("turquoise", "darkturquoise"),
            ("slateblue", "darkslateblue"),
            ("royalblue", "darkblue"),
        ]

        # Defining the x-axis
        iterations = len(summary_df)
        t = np.linspace(0.0, iterations - 1, iterations)
        # Set x-axis title
        fig.update_xaxes(title_text="Timestep (-)", row=num_rows, col=1)

        for i, column_name in enumerate(summary_df.columns):
            # Add trace of the primary plot
            fig.add_trace(
                go.Scatter(
                    name=column_name,
                    x=t,
                    y=np.array(summary_df[column_name]),
                    mode="lines",
                    line=dict(width=1, color=colors[i][0]),
                ),
                row=i + 1,
                col=1,
                secondary_y=False,
            )

            unit = opts[column_name]["unit"]
            range = opts[column_name]["range"]

            # Applying range to y-axis if it is specified
            if range is not None:
                fig.update_yaxes(
                    title_text=f"<b>{column_name}</b> { unit}",
                    range=range,
                    row=i + 1,
                    col=1,
                )
            else:
                fig.update_yaxes(
                    title_text=f"<b>{column_name}</b> { unit}", row=i + 1, col=1
                )

            # Creating auxiliary plot if it specified
            if opts[column_name]["secondary_y"] is not None:
                suffix = ""
                arr = summary_df[column_name]

                if opts[column_name]["secondary_y"] == "moving_average":
                    suffix = "avg"
                    arr = np.array(arr.rolling(24).mean())

                elif opts[column_name]["secondary_y"] == "cumulative":
                    suffix = "cumulative"
                    arr = np.cumsum(np.array(arr))

                fig.add_trace(
                    go.Scatter(
                        name=f"{column_name}_{suffix}",
                        x=t,
                        y=arr,
                        mode="lines",
                        line=dict(width=2, color=colors[i][1]),
                    ),
                    row=i + 1,
                    col=1,
                    secondary_y=True,
                )

                # Updating y-axis with range only if it's the moving average option
                if range is not None and (
                    opts[column_name]["secondary_y"] == "moving_average"
                ):
                    fig.update_yaxes(
                        title_text=f"<b>{column_name}_{suffix}</b> { unit}",
                        range=range,
                        row=i + 1,
                        col=1,
                        secondary_y=True,
                    )
                else:
                    fig.update_yaxes(
                        title_text=f"<b>{column_name}_{suffix}</b> { unit}",
                        row=i + 1,
                        col=1,
                        secondary_y=True,
                    )

        fig.update_xaxes(nticks=50)
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Courier New, monospace", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        )

        fig.update_layout(title_text=title)

        pyo.plot(fig, filename=plot_filename)

    def log(
        self, summary_df: pd.DataFrame, suffix: str, is_summary: bool, agent: Agent
    ) -> None:
        """"Utility function to log all the relevant data.
        It logs:
        All the relevant data present in summary_df, all the relevant data from the agent, 

        Args:
            summary_df (pd.DataFrame):  A pandas DataFrame containing a summary of the important parameters of the session
            (to be defined depending on the agent and the use case)
            suffix (str): Suffix to be added to the logging path
            is_summary (bool): Boolean flag to know whether we are logging a single episode or the entire session
            agent (Agent): The agent that was trained/tested/assessed

        Returns:
            _type_: _description_
        """

        summary_df.to_csv(
            f"{self.RESULT_PATH}/experiments_csv/experiments_results_{suffix}.csv"
        )

        ## saving parameters of environment

        log_dict = agent.log_dict()

        ## concatenate the two dicts
        log_dict = {**log_dict, **agent.env.log_dict()}

        log_dict["num_episodes"] = self.num_episodes
        log_dict["num_iterations"] = self.num_iterations

        #log_dict["final_reward"] = np.array(summary_df["Reward"]).cumsum()[-1]
        #log_dict["final_cumulative_heating"] = np.array(summary_df["Heating"]).cumsum()[
        #    -1
        #]

        ## logging pmv percentages
        pmv_df = self.pmv_percentages(np.array(summary_df["PMV"])).to_dict()

        log_dict["pmvs"] = {
            pmv_df["intervals"][i]: pmv_df["ranges"][i]
            for i in pmv_df["intervals"].keys()
        }

        # writing the json
        f = open(f"{self.RESULT_PATH}/env_params_{self.time}.json", "w")
        f.write(json.dumps(log_dict, indent=True))
        f.close()

        ## save agent model weights
        agent.save(
            directory=f"{self.RESULT_PATH}/model_weights", filename=f"torch_ep_{suffix}"
        )

        return summary_df

    def plot_and_logging(
        self,
        summary_df: pd.DataFrame,
        agent: Agent,
        episode_num: int,
        is_summary: bool,
        opts: Dict[str, Dict[str, Any]],
    ):
        """Main method of the logger that takes care of saving/logging all the data and plotting
        relevant summaries of the data.

        Args:
            summary_df (pd.DataFrame): Should contain all the relevant data needed to log and plot.
            agent (Agent): The agent that was trained/tested/assessed
            episode_num (int): The current episode number of the session (the logger can be called for every episode )
            is_summary (bool): Boolean flag to make sure the titles make sense
            opts (Dict[str, Dict[str, Any]]): Dictionary defining the plotting parameters.
            We refer to the documentation of plot() for further details.
        """

        ## defining the suffix to be used for the filenames
        suffix = str(episode_num + 1)
        if is_summary:
            suffix = "summary"

        ## defining filenames and titles
        plot_filename = f"{self.RESULT_PATH}/plots/summary/results_{suffix}.html"
        plot_title = (
            f"Episode Number {suffix} with α = {agent.env.alpha} and β = {agent.env.beta}"
            if not (is_summary)
            else f"Summary with α = {agent.env.alpha} and β = {agent.env.beta}"
        )

        self.plot(
            summary_df=summary_df,
            plot_filename=plot_filename,
            title=plot_title,
            opts=opts,
        )

        ## PLOTTING PMV INTERVALS
        plot_title = f"Number of hours the algorithm spent in different PMV intervals"
        plot_title = plot_title + " (Total)" if is_summary else ""

        ## only keep pmv when occupancy > 0
        Logger.plot_pmv_percentages(
            pmv=np.array(summary_df[summary_df["Occ"] > 0]["PMV"]),
            savepath=f"{self.RESULT_PATH}/plots/pmv_categories",
            filename=f"PMV_Categories_{suffix}",
            plot_title=plot_title,
        )

        ## Logging part
        self.log(
            summary_df=summary_df, suffix=suffix, is_summary=is_summary, agent=agent
        )

        return

