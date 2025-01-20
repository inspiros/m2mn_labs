from typing import Callable, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab1.mab import UCBBandit
from mab import EpsilonBandit

DEFAULT_REALIZATIONS: int = 100


def manipulation_loop(realizations: int = 1,
                      steps: int = 1000,
                      *,
                      pre_hook: Optional[Callable[[EpsilonBandit], None]] = None,
                      return_selections: bool = False,
                      mab_class: type = EpsilonBandit,
                      **mab_kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Manipulation Loop.

    Parameters:
        realizations (int): Number of realizations.
        steps (int): Number of steps.
        pre_hook (Callable[[EpsilonBandit], None]): Pre-run hook function.
        return_selections (bool): Whether to return selections.
        mab_class (type): MAB class. Defaults to `EpsilonBandit`.
        **mab_kwargs (dict): Additional arguments passed to `mab`.
    """
    avg_rewards = np.zeros(steps)
    selections = np.zeros(mab_kwargs['k']) if return_selections else None
    for i in range(realizations):
        mab = mab_class(**mab_kwargs)
        if pre_hook is not None:
            pre_hook(mab)
        rewards = mab.run(steps, return_rewards=True)
        # update long-term averages among episodes
        avg_rewards = avg_rewards + (rewards - avg_rewards) / (i + 1)
        if return_selections:
            selections = selections + (mab.k_n - selections) / (i + 1)
    if return_selections:
        return avg_rewards, selections
    return avg_rewards


def manipulation_1(k: int, steps: int) -> None:
    print('----- Manipulation 1 -----')
    # part 1 & 2
    fig, ax = plt.subplots()
    mu = np.random.normal(0, 1, k)
    for realizations in [1, 10, 100, 1000]:
        rewards = manipulation_loop(realizations=realizations, steps=steps,
                                    k=k, eps=0.1, mu=mu)
        ax.plot(rewards, label=f'$\\epsilon={0.1}$, {realizations} realizations')
    ax.legend()
    ax.grid()
    ax.set_xlim(0, steps)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Average Rewards')
    fig.tight_layout()
    plt.show()

    # part 3
    fig, ax = plt.subplots()
    mu = np.random.normal(0, 1, k)
    for eps in [.1, .01, 0]:
        rewards = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                    k=k, eps=eps, mu=mu)
        ax.plot(rewards, label=f'$\\epsilon={eps}$' if eps else 'greedy')
    ax.legend()
    ax.grid()
    ax.set_xlim(0, steps)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Average $\\epsilon$-Greedy and Greedy Rewards')
    fig.tight_layout()
    plt.show()
    print()


def manipulation_2(k: int, steps: int) -> None:
    print('----- Manipulation 2 -----')
    # part ?
    eps_configs = [.1, .01, 0]
    all_selections = []

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    bar_width = .25
    for i, eps in enumerate(eps_configs):
        rewards, selections = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                                return_selections=True,
                                                k=k, eps=eps, mu='sequence')
        all_selections.append(selections)

        label = f'$\\epsilon={eps}$' if eps else 'greedy'
        axes[0].plot(rewards, label=label)
        axes[1].bar(np.linspace(0, k - 1, k) + bar_width * (i - (len(eps_configs) - 1) // 2),
                    selections, width=bar_width, label=label, zorder=3)
    opt_per = np.asarray(all_selections) / steps * 100
    df = pd.DataFrame(opt_per,
                      index=[f'$\\epsilon={eps}$' for eps in eps_configs],
                      columns=[f'$a={a}$' for a in range(k)])
    print(df.to_markdown())

    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlim(0, steps)
    axes[0].set_ylim(0, axes[0].get_ylim()[1])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title(f'Average Rewards')

    axes[1].legend()
    axes[1].yaxis.grid(True, zorder=0)
    axes[1].set_xticks(np.arange(0, k, 1))
    axes[1].set_xlabel('$a\\in\\mathcal{A}$')
    axes[1].set_ylabel('$N(a)$')
    axes[1].set_title('Actions Selected')

    fig.tight_layout()
    fig.suptitle('$\\mu$="sequence"')
    plt.show()
    print()


def manipulation_3(k: int, steps: int) -> None:
    print('----- Manipulation 3 -----')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # part 1
    n = 200
    for beta in np.linspace(0, 1, 6):
        axes[0].plot([1 / (1 + n * beta) for n in range(n)], label=f'$\\beta={beta:.1f}$')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('$N$')
    axes[0].set_ylabel('$\\epsilon=\\frac{1}{1 + N.\\beta}$')
    axes[0].set_title(f'$\\epsilon$ curve with different $\\beta$')

    # part 3
    mu = np.random.normal(0, 1, k)
    for beta in np.linspace(0, 1, 6):
        rewards = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                    k=k, eps='decreasing', beta=beta, mu=mu)
        axes[1].plot(rewards, label=f'$\\beta={beta:.1f}$')
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlim(0, steps)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title(f'Average rewards with different $\\beta$')
    fig.tight_layout()
    plt.show()
    print()


def manipulation_4(k: int, steps: int) -> None:
    print('----- Manipulation 4 -----')

    def optimistic_init(mab: EpsilonBandit) -> None:
        # optimistic initialization
        mab.eps = mab._init_eps = 0
        mab.k_reward = np.repeat(5., mab.k)
        mab.k_n = np.ones_like(mab.k_n)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab10')
    mu = np.random.normal(0, 1, k)
    for i, eps in enumerate([.1, .01, 0]):
        rewards = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                    k=k, eps=eps, mu=mu)
        ax.plot(rewards, linestyle='--', color=cmap(i), label=f'$\\epsilon={eps}$ without OI')
        rewards = manipulation_loop(realizations=DEFAULT_REALIZATIONS,
                                    pre_hook=optimistic_init, steps=steps,
                                    k=k, eps=0, mu=mu)
        ax.plot(rewards, linestyle='-', color=cmap(i), label=f'$\\epsilon={eps}$ with OI')
    ax.legend()
    ax.grid()
    ax.set_xlim(0, steps)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Average Rewards with and without Optimistic Initialization')
    fig.tight_layout()
    plt.show()
    print()


def manipulation_5(k: int, steps: int) -> None:
    print('----- Manipulation 5 -----')
    eps_configs = [.1, .01, 0]
    c_configs = [2, 1, 0]
    all_selections = []
    labels = []

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    cmap = plt.get_cmap('tab10')
    alpharize = lambda color, alpha: (*color[:-1], alpha)
    bar_width = 0.1
    for i, eps in enumerate(eps_configs):
        rewards, selections = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                                return_selections=True,
                                                k=k, eps=eps, mu='sequence')
        all_selections.append(selections)

        label = f'{EpsilonBandit.__name__}($\\epsilon={eps}$)'
        labels.append(label)
        axes[0].plot(rewards, color=cmap(i), linestyle='--', label=label)
        axes[1].bar(np.linspace(0, k - 1, k) + bar_width * (i - (len(eps_configs) + len(c_configs) - 1) // 2),
                    selections, color=alpharize(cmap(i), 0.5), width=bar_width, label=label, zorder=3)

    for i, c in enumerate(c_configs):
        rewards, selections = manipulation_loop(realizations=DEFAULT_REALIZATIONS, steps=steps,
                                                return_selections=True,
                                                mab_class=UCBBandit,
                                                k=k, c=c, mu='sequence')
        all_selections.append(selections)

        label = f'{UCBBandit.__name__}($c={c}$)'
        labels.append(label)
        axes[0].plot(rewards, color=cmap(i), linestyle='-', label=label)
        axes[1].bar(np.linspace(0, k - 1, k) + bar_width * (
                i - (len(eps_configs) + len(c_configs) - 1) // 2 + len(eps_configs)),
                    selections, color=cmap(i), width=bar_width, label=label, zorder=3)

    opt_per = np.asarray(all_selections) / steps * 100
    df = pd.DataFrame(opt_per,
                      index=labels,
                      columns=[f'$a={a}$' for a in range(k)])
    print(df.to_markdown())

    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlim(0, steps)
    axes[0].set_ylim(0, axes[0].get_ylim()[1])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title(f'Average Rewards with and without UCB')

    axes[1].legend()
    axes[1].yaxis.grid(True, zorder=0)
    axes[1].set_xticks(np.arange(0, k, 1))
    axes[1].set_xlabel('$a\\in\\mathcal{A}$')
    axes[1].set_ylabel('$N(a)$')
    axes[1].set_title('Actions Selected')

    fig.tight_layout()
    plt.show()
    print()


def main(k: int = 10, steps: int = 2000):
    # manipulation_1(k=k, steps=steps)
    # manipulation_2(k=k, steps=steps)
    # manipulation_3(k=k, steps=steps)
    # manipulation_4(k=k, steps=steps)
    manipulation_5(k=k, steps=steps)


if __name__ == '__main__':
    main()
