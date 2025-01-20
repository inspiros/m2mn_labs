import math

import numpy as np
from typing import Optional, Union

__all__ = ['GreedyBandit', 'EpsilonBandit', 'UCBBandit']


class _BaseBandit(object):
    r"""Base class for Multi-Armed Bandits algorithms."""
    def __init__(self, k: int, mu: Union[str, np.ndarray] = 'random') -> None:
        if k <= 0:
            raise ValueError('k must be positive')
        self.k = k
        self.n = 0  # step count
        self.k_n = np.zeros(k)  # step count for each arm
        self.mean_reward = 0  # overall mean reward
        self.k_reward = np.zeros(k)  # mean reward for each arm
        if isinstance(mu, str):
            if mu == 'random':
                # draw means from probability distribution
                self.mu = np.random.normal(0, 1, k)
            elif mu == 'sequence':
                # increase the mean for each arm by one
                self.mu = np.linspace(0, k - 1, k)
            else:
                raise ValueError('mu must be "random" or "sequence"')
        else:
            # user-defined averages
            self.mu = np.asarray(mu)

    def reset(self) -> None:
        # resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.k_reward = np.zeros(self.k)

    def select_arm(self) -> int:
        r"""Select an arm to be pulled"""
        raise NotImplementedError

    def pull(self, a: int) -> None:
        r"""Pull an arm."""
        # random generation of reward
        reward = np.random.normal(self.mu[a], 1)
        # update counts
        self.n += 1
        self.k_n[a] += 1
        # update overall mean reward
        self.mean_reward += (reward - self.mean_reward) / self.n
        # update reward for a_k
        self.k_reward[a] += (reward - self.k_reward[a]) / self.k_n[a]

    def run(self, steps: int, return_rewards: bool = False) -> Optional[np.ndarray]:
        r"""Run a simulation for one bandit realization."""
        rewards = np.zeros(steps) if return_rewards else None
        for i in range(steps):
            self.pull(self.select_arm())
            if return_rewards:
                rewards[i] = self.mean_reward
        if return_rewards:
            return rewards


class GreedyBandit(_BaseBandit):
    r"""
    Greedy k-Bandits class.

    Parameters:
        k (int): number of arms
        mu: The average rewards for each of the k-arms.
            Set to "random" for the rewards to be selected from
            a normal distribution with mean = 0.
            Set to "sequence" for the means to be ordered from
            0 to k-1.
            Or pass a list or array of length k for user-defined
            values.
    """

    def __init__(self, k: int, mu: Union[str, np.ndarray] = 'random') -> None:
        super().__init__(k, mu)

    def select_arm(self) -> int:
        r"""Select an arm using greedy policy."""
        if self.n == 0:
            a = np.random.choice(self.k)
        else:
            a = np.argmax(self.k_reward)
        return a


class EpsilonBandit(_BaseBandit):
    r"""
    ϵ-greedy k-Bandits class.

    Parameters:
        k (int): number of arms.
        eps (float, str): probability of random action 0 < eps < 1
            or "decreasing" for decaying probability behavior.
        beta (float): decay parameter. Defaults to 1.
        mu: The average rewards for each of the k-arms.
            Set to "random" for the rewards to be selected from
            a normal distribution with mean = 0.
            Set to "sequence" for the means to be ordered from
            0 to k-1.
            Or pass a list or array of length k for user-defined
            values.
    """

    def __init__(self, k: int, eps: Union[float, str] = 0., beta: float = 1.,
                 mu: Union[str, np.ndarray] = 'random') -> None:
        super().__init__(k, mu)
        self.eps = self._init_eps = eps
        self.beta = beta

    def reset(self) -> None:
        super().reset()
        self.eps = self._init_eps

    def select_arm(self) -> int:
        r"""Select an arm using ϵ-greedy policy."""
        if self._init_eps == 'decreasing':
            self.eps = 1 / (1 + self.n * self.beta)
        else:
            self.eps *= self.beta
        if (self.eps == 0 and self.n == 0) or np.random.rand() < self.eps:
            # randomly select an action
            a = np.random.choice(self.k)
        else:
            # take greedy action
            a = np.argmax(self.k_reward)
        return a


class UCBBandit(_BaseBandit):
    r"""
    Upper-Confidence-Bound k-Bandits class.

    Parameters:
        k (int): number of arms.
        c (float): factor controlling the degree of exploration.
        mu: The average rewards for each of the k-arms.
            Set to "random" for the rewards to be selected from
            a normal distribution with mean = 0.
            Set to "sequence" for the means to be ordered from
            0 to k-1.
            Or pass a list or array of length k for user-defined
            values.
    """

    def __init__(self, k: int, c: float,
                 mu: Union[str, np.ndarray] = 'random') -> None:
        super().__init__(k, mu)
        if c < 0:
            raise ValueError('c must be non negative')
        self.c = c

    def select_arm(self) -> int:
        r"""Select an arm using UCB policy."""
        if self.c == 0:
            # take greedy action
            a = np.argmax(self.k_reward)
        else:
            if np.any(self.k_n == 0):
                # randomly select an action that has not been tested once
                a = np.random.choice(np.where(self.k_n == 0)[0])
            else:
                # ucb
                a = np.argmax(self.k_reward + self.c * np.sqrt(np.log(self.n) / self.k_n))
        return a
