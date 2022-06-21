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


class DQNAgent(Agent):
    def __init__(
        self,
        env: Environment,
        memory_size: int = 1000,
        batch_size: int = 32,
        actor_update: int = 10,
        target_update: int = 100,
        epsilon_decay: float = 1 / 20000,
        lr: float = 1e-3,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.0,
        gamma: float = 0.99,
        inside_dim: int = 128,  ## dimension of the hidden layers of the network
        num_hidden_layers: int = 1,
        seed: int = 778,
    ):
        """Deep Q-Learning Agent. Made to interact with DiscreteEnvironment. Inherits from Agent class.

        Attributes:
            epsilon(float): Parameter that defines the level of exploration of the agent. We refer to the definition
            of epsilon-greedy action selection (https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/).
            replay_buffer(ReplayBuffer): Replay buffer to store past experiences, experiences that will be used to train the DQN.
            transition: A list containing the current state, the action taken and the next state of the agent.
            dqn(Network): Neural network approximating the Q-function.  

        Args:
            env (Environment): Environment inheriting from the Environment class.
            memory_size (int, optional): Size of replay buffer memory. Defaults to 1000.
            batch_size (int, optional): Batch size for sampling. Defaults to 32.
            actor_update (int, optional): How many environment steps between the DQN training epoch. Defaults to 10.
            target_update (int, optional): How many environment steps between the target DQN hard update. Defaults to 100.
            epsilon_decay (float, optional): Step size to decrease epsilon. Defaults to 1/20000.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            max_epsilon (float, optional): Maximum and starting value of epsilon. Defaults to 1.0.
            min_epsilon (float, optional): Minimum and last value of epsilon. Defaults to 0.0.
            gamma (float, optional): Discount factor of the Bellman equation Defaults to 0.99.
            inside_dim (int, optional): DQN network hidden layer size. Defaults to 128.
            num_hidden_layers(int,optional): DQN number of hidden layers. Defaults to 1.
            seed (int, optional): Seed to seed the agent. Defaults to 778.
        """

        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.lr = lr
        self.target_update = target_update
        self.actor_update = actor_update
        self.gamma = gamma
        self.seed = seed
        self.inside_dim = inside_dim
        self.num_hidden_layers = num_hidden_layers
        self.opts={}

        ## seeding the agent
        self.seed_agent(self.seed)

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## dimensions for the network
        obs_dim: int = self.env.observation_dim
        discrete_action_dim: int = self.env.discrete_action_dim

        ## setting up memory
        self.replay_buffer = ReplayBuffer(obs_dim, self.memory_size, self.batch_size)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim,
            discrete_action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        self.dqn_target = Network(
            obs_dim,
            discrete_action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """We refer to the Agent class docstring."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            # selected_action = np.random.choice(self.env.discrete_action_dim, 1)[0]
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state.T).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """We refer to the Agent class docstring."""
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)

        return next_state, reward, done, info

    def reset(self) -> Agent:
        """We refer to the Agent class docstring."""

        self.epsilon = self.max_epsilon

        self.seed_agent(self.seed)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_dim, self.memory_size, self.batch_size
        )

        # networks: dqn, dqn_target
        self.dqn = Network(
            self.env.observation_dim,
            self.env.discrete_action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        self.dqn_target = Network(
            self.env.observation_dim,
            self.env.discrete_action_dim,
            inside_dim=self.inside_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        return self

    def save(self, filename, directory):
        """We refer to the Agent class docstring."""
        torch.save(self.dqn.state_dict(), "%s/%s_dqn.pth" % (directory, filename))
        torch.save(
            self.dqn_target.state_dict(), "%s/%s_dqn_target.pth" % (directory, filename)
        )

    def load(self, filename, directory):
        """We refer to the Agent class docstring."""
        state_dict = torch.load("%s/%s_dqn.pth" % (directory, filename))
        self.dqn.load_state_dict(state_dict)
        self.dqn_target.load_state_dict(
            torch.load("%s/%s_dqn_target.pth" % (directory, filename))
        )

    def log_dict(self) -> Dict[str, Any]:
        """We refer to the Agent class docstring."""
        log_dict = {
            "is_test": self.is_test,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "actor_update": self.actor_update,
            "epsilon_decay": self.epsilon_decay,
            "max_epsilon": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "inside_dim": self.inside_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "seed": self.seed,
        }

        return log_dict

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
                agent_name="DQN_Agent",
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
            "Epsilon": {"secondary_y": None, "range": None, "unit": "(-)",},
            "Loss": {"secondary_y": None, "range": None, "unit": "(-)",},
        }

        summary_df: pd.DataFrame = pd.DataFrame()

        epsilons = []
        losses = []
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

            ## to keep track of number of updates and update target network accordingly
            update_cnt = 0

            for i in range(num_iterations):

                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                if i % 1000 == 0:
                    print(f"Iteration{i}")

                ## keeping track of the value we've seen
                rewards.append(reward)
                ## because our actions are discrete values
                actions.append(self.env.action_to_temp[action])
                pmv.append(info["pmv"][0])
                d = self.env.observation_to_dict(next_state)
                tair.append(d["Tair"][0])
                heat = d["Qheat"][0]
                qheat.append(heat)
                occ.append(d["Occ"][0])

                state = next_state

                # if training is ready
                if not (self.is_test) and (len(self.replay_buffer) >= self.batch_size):

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon,
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    )
                    epsilons.append(self.epsilon)

                    # if DQN update is needed
                    if update_cnt % self.actor_update == 0:
                        loss = self.update_model()
                        losses.append(loss)

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                    update_cnt += 1

            ## creating lower and upper index to be used below to slice through lists of attributes
            lower = episode_num * num_iterations
            upper = (episode_num + 1) * num_iterations

            # epsilons and losses have different length since they only start to fill up when
            # the replay buffer is full (i.e. we can train) thus we must pad/extend the length.
            # this is because plotting requires same length lists.
            if not (self.is_test):
                len_difference = len(tair) - len(epsilons)
                pad_epsilon = [epsilons[0] for i in range(len_difference)]
                epsilons = pad_epsilon + epsilons

                temp_losses = [
                    loss for loss in losses for _ in range(self.actor_update)
                ]
                len_difference = len(tair) - len(temp_losses)
                pad_losses = [0 for i in range(len_difference)]
                temp_losses = pad_losses + temp_losses

            columns = {
                "Tair": tair[lower:upper],
                "Tset": actions[lower:upper],
                "PMV": pmv[lower:upper],
                "Heating": qheat[lower:upper],
                "Reward": rewards[lower:upper],
                "Occ": occ[lower:upper],
            }

            if not (self.is_test):
                columns["Loss"] = temp_losses[lower:upper]
                columns["Epsilon"] = epsilons[lower:upper]

            summary_df = pd.DataFrame(columns)

            summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

            if log:
                logger.plot_and_logging(
                    summary_df=summary_df,
                    agent=self,
                    episode_num=episode_num,
                    is_summary=False,
                    opts=self.opts,
                )

        # Summary that contatenates all episodes together for a complete overview of the training

        ## As above, extend the length of the loss array such that it is of correct size for plotting
        if not (self.is_test):
            temp_losses = [loss for loss in losses for _ in range(self.actor_update)]
            len_difference = len(tair) - len(temp_losses)
            pad_losses = [0 for i in range(len_difference)]
            temp_losses = pad_losses + temp_losses

        columns = {
            "Tair": tair,
            "Tset": actions,
            "PMV": pmv,
            "Heating": qheat,
            "Reward": rewards,
            "Occ": occ,
        }

        if not (self.is_test):
            columns["Loss"] = temp_losses
            columns["Epsilon"] = epsilons

        summary_df = pd.DataFrame(columns)

        summary_df["Reward"] = summary_df["Reward"].apply(lambda x: float(x[0]))

        if log and num_episodes > 1:
            logger.plot_and_logging(
                summary_df=summary_df,
                agent=self,
                episode_num=num_episodes,
                is_summary=True,
                opts=self.opts,
            )

            # self.save(
            # directory=f"{logger.RESULT_PATH}/model_weights", filename=f"torch_ep_summary")

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
        """We refer to the Agent class docstring."""

        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(seed)

        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        return

    # DQN specific
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.replay_buffer.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # DQN specific
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # current estimation of Q value
        curr_q_value = self.dqn(state).gather(1, action)
        # estimated future Q value
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        # Bellman equation (in theory, it is equal to current Q value if Q is optinal)
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    # DQN Specific
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


# Helper classes


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_dim = obs_dim
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs.reshape(self.obs_buf[self.ptr].shape)
        self.next_obs_buf[self.ptr] = next_obs.reshape(self.obs_buf[self.ptr].shape)
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, inside_dim: int = 128, num_hidden_layers=1
    ):
        """Initialization."""
        super(Network, self).__init__()

        # first layer
        self.layers = nn.ModuleList([nn.Linear(in_dim, inside_dim)])
        self.layers.append(nn.ReLU())

        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(inside_dim, inside_dim))
            self.layers.append(nn.ReLU())

        # last layer
        self.layers = self.layers.append(nn.Linear(inside_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        for layer in self.layers:
            x = layer(x)

        return x
