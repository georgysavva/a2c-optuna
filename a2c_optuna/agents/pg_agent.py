from typing import Optional, Sequence

import einops
import numpy as np
import torch
from torch import nn

from a2c_optuna.infrastructure import pytorch_util as ptu
from a2c_optuna.networks.critics import ValueCritic
from a2c_optuna.networks.policies import MLPPolicyPG


class PGAgent(nn.Module):

    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        value_td_lambda: float,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.value_td_lambda = value_td_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        value_bootstraps: np.ndarray,
        dones: np.ndarray,
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """
        assert self.critic is not None, "Critic must be used to estimate advantages"

        values = self.critic.predict(obs)
        advantages: np.ndarray = self._estimate_advantage(
            self.gae_lambda, rewards, values, value_bootstraps, dones
        )
        advantages_for_returns: np.ndarray = self._estimate_advantage(
            self.value_td_lambda, rewards, values, value_bootstraps, dones
        )
        returns = values + advantages_for_returns

        advantages = einops.rearrange(advantages, "t env -> (t env)")
        actions = einops.rearrange(actions, "t env ac -> (t env) ac")
        obs = einops.rearrange(obs, "t env ob -> (t env) ob")
        returns = einops.rearrange(returns, "t env -> (t env)")
        # normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = (advantages - mean) / (std + 1e-12)
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # update the PG actor/policy network once using the advantages

        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            assert self.baseline_gradient_steps is not None
            for _ in range(self.baseline_gradient_steps):
                critic_step_info = self.critic.update(obs, returns)
                for key, value in critic_step_info.items():
                    critic_info[key] = critic_info.get(key, 0) + value
            for key in critic_info:
                critic_info[key] /= self.baseline_gradient_steps
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        td_lambda: float,
        rewards: np.ndarray,
        values: np.ndarray,
        value_bootstraps: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        # run the critic and use it as a baseline
        batch_size = values.shape[0]

        advantages = np.zeros_like(values)
        last_advantage = np.zeros(values.shape[-1])
        for i in reversed(range(batch_size)):
            # recursively compute advantage estimates starting from timestep T.
            if i == batch_size - 1:
                next_values = np.zeros_like(values.shape[-1])
            else:
                next_values = values[i + 1]

            delta = (
                rewards[i]
                + self.gamma * value_bootstraps[i]
                + self.gamma * next_values * (1 - dones[i])
                - values[i]
            )

            last_advantage = (
                self.gamma * td_lambda * last_advantage * (1 - dones[i]) + delta
            )
            advantages[i] = last_advantage

        return advantages

    def _discounted_return(self, rewards: np.ndarray) -> np.ndarray:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        gammas = self.gamma ** np.arange(len(rewards))
        discounted_rewards = gammas * rewards
        result = discounted_rewards.sum() * np.ones(len(rewards))
        return result

    def _discounted_reward_to_go(self, rewards: np.ndarray) -> np.ndarray:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        gammas = self.gamma ** np.arange(len(rewards))
        discounted_rewards = gammas * rewards
        rewards_to_go = np.cumsum(discounted_rewards[::-1])[::-1]
        result = rewards_to_go / gammas
        return result
