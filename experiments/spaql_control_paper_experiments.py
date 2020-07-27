#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:24:51 2020

This file generates the images for the Control with Adaptive Q-Learning paper.
"""

import os
import sys
import pickle
from copy import deepcopy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt_style = "fivethirtyeight"
fontsize = 'x-small'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# The code in this file is highly paralelizable
from joblib import Parallel, delayed
from functools import partial
n_jobs = 4

# Import the agents: Random
from agents import RandomAgent

# Import the envs: Pendulum, CartPole
from pendulum_agent import PendulumAgent, PendulumAgentTerminalState, \
                           PendulumEnv, MultiPartitionPendulum
from cartpole_agent import CartPoleAgent, CartPoleEnv, MultiPartitionCartPole, \
                           CartPoleAgentTerminalState, CartPoleRandomAgent

# Experimental setup
from spaql_experiment import start

from plot_rl_experiment import plot_rl_exp

# For the statistical power analysis
from test_RL_difference import welch_test

def load_data(log_dir):
    """
    Returns the average rewards along training, along with the number of arms.

    Parameters
    ----------
    log_dir : str
        The directory where agentData.pk is stored.

    Returns
    -------
    The data in the agentData.pk file.

    """
    with open(f"{log_dir}/agentData.pk", 'rb') as f:
        data = pickle.load(f)
    return data

def scaling_wrapper(**kwargs):
    """
    Wrapper function to start, used when tuning the value of the scaling parameter.
    Handles caching automatically by checking if the agentData.pk file already
    exists in folder agent_log_dir.

    Parameters
    ----------
    **kwargs : dict
        The parameters to be passed to start(...).

    Returns
    -------
    array
        An array with the average cumulative reward obtained, and corresponding
        standard deviation.

    """
    try:
        agent_log_dir = kwargs["agent_log_dir"]
        with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
            data = pickle.load(f)
            rewards = data["rewards"]
    except:
        now = time.time()
        print("Actually started training...")
        rewards, agents = start(**kwargs)
        print(f"Time elapsed: {time.time() - now}")
    return np.array([np.average(rewards[-1,:]), np.std(rewards[-1,:])])

def start_wrapper(**kwargs):
    """
    Wrapper function to start.
    Handles caching automatically by checking if the agentData.pk file already
    exists in folder agent_log_dir.

    Parameters
    ----------
    **kwargs : dict
        The parameters to be passed to start(...).

    Returns
    -------
    rewards : array
        Evolution of cumulative reward along training.
    agents : list
        The agents trained

    """
    try:
        agent_log_dir = kwargs["agent_log_dir"]
        with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
            data = pickle.load(f)
            rewards = data["rewards"]
            agents = data["agents"]
    except:
        now = time.time()
        rewards, agents = start(**kwargs)
        print(f"Time elapsed: {time.time() - now}")
    return rewards, agents

if __name__ == "__main__":
    input("""WARNING: Running this file as it is will duplicate the experiments in the "Control with AQL" paper.
These experiments take a long time to run, and the results occupy some GB.

Press Enter to continue...""")
    #
    # Set training parameters
    #
    train_iter = 100  # number of training iterations
    epLen = 200        # episode length
    nExp = 20          # number of experiments to run
    
    n_jobs = 4
    
    #
    # Set algorithm parameters
    #
    M = 0.1
    u = 2
    d = 0.8
    
    lam = 1.2
    
    """
    Pendulum experiments
    """
    if not os.path.exists("results"):
        os.mkdir("results")
    
    root_log_dir = "results/ControlAQLPaperResults" # Directory where everything will be logged
    figures_dir = root_log_dir + "/figures"
    scaling_dir = root_log_dir + "/scaling"
    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
        os.mkdir(figures_dir)
        os.mkdir(scaling_dir)
    
    prints = False
    
    # Random seed
    seed = None

    scaling_values = np.array([0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.75, 3, 4])/5*epLen
    
    # Plot the effect of the scaling parameter
    with plt.style.context(plt_style):
        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(8,4))
        # fig.suptitle("Effect of scaling parameter ")
        # axs[0].set_ylim((None, None))
    
    log_dir = root_log_dir + "/scaling/pendulum"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # This loop finds the best scaling value
    opt_mp_scaling_pendulum = []
    opt_sp_scaling_pendulum = []
    opt_spts_scaling_pendulum = []
    for i, scale_reward in enumerate([0, 1]):
        # Set the environment
        env = PendulumEnv(scale_reward=scale_reward)
        env.env.seed(seed)
            
        env_name = f"{scale_reward}_Pendulum"
        
        """
        Baselines: Random Agent
        """
        scaling = ""
        agent_class = RandomAgent
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
        rewardsRandomSc, sc_random_agent_list = start_wrapper(agent_log_dir=agent_log_dir,
                nExp=nExp,
                agentClass=agent_class,
                classKwargs={
                    'epLen': epLen,
                    'numIters': None,
                    },
                experimentKwargs={
                    'train_iter': train_iter,
                    'nExp': nExp,
                    'env': env,
                    'video': False,
                    'debug': prints,
                    'sps': False,
                    },
                plot=False)

        agent_class = MultiPartitionPendulum
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        rewards_mp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                    nExp=nExp,
                                                    agentClass=agent_class,
                                                    classKwargs={
                                                        'epLen': epLen,
                                                        'numIters': None,
                                                        'scaling': scaling
                                                        },
                                                    experimentKwargs={
                                                        'train_iter': train_iter,
                                                        'nExp': nExp,
                                                        'env': env,
                                                        'video': False,
                                                        'debug': prints,
                                                        'sps': False,
                                                        },
                                                    plot=False) for scaling in scaling_values)
        
        # SPAQL
        agent_class = PendulumAgent
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
        rewards_sp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'scaling': scaling,
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps_es': True,
                                                    'u': u,
                                                    'search_type': 'heuristic',
                                                    'd': d,
                                                    },
                                                plot=False) for scaling in scaling_values)
        
        # SPAQL-TS
        agent_class = PendulumAgentTerminalState
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
        rewards_spts_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'scaling': scaling,
                                                    'lam': lam
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps_es': True,
                                                    'u': u,
                                                    'search_type': 'heuristic',
                                                    'd': d,
                                                    },
                                                plot=False) for scaling in scaling_values)
        
        rewards_mp_sc = np.vstack(rewards_mp_sc)
        rewards_sp_sc = np.vstack(rewards_sp_sc)
        rewards_spts_sc = np.vstack(rewards_spts_sc)
        
        # [20, 160]
        opt_mp_scaling_pendulum.append(scaling_values[np.argmax(rewards_mp_sc[:,0])])
        # [4, 10]
        opt_sp_scaling_pendulum.append(scaling_values[np.argmax(rewards_sp_sc[:,0])])
        # [0.4, 0]
        opt_spts_scaling_pendulum.append(scaling_values[np.argmax(rewards_spts_sc[:,0])])
        print(f"The best scaling value for the Multi Partition Agents was {scaling_values[np.argmax(rewards_mp_sc[:,0])]}")
        print(f"The best scaling value for the Single Partition Agents was {scaling_values[np.argmax(rewards_sp_sc[:,0])]}")
        print(f"The best scaling value for the Single Partition TS Agents was {scaling_values[np.argmax(rewards_spts_sc[:,0])]}")
        
        # Now that we have the best scaling values for single and multipartion,
        # it is time to plot them. We need average cumulative reward with
        # 95% confidence interval, and number of arms
        
        mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    MultiPartitionPendulum.__name__,
                                                    env_name,
                                                    scaling_values[np.argmax(rewards_mp_sc[:,0])])
        mp_data = load_data(mp_agent_log_dir)
        rewards_mp = mp_data["rewards"]
        arms_mp = mp_data['arms']
        agents_mp = mp_data['agents']
        mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
        
        sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    PendulumAgent.__name__,
                                                    env_name,
                                                    scaling_values[np.argmax(rewards_sp_sc[:,0])])
        sp_data = load_data(sp_agent_log_dir)
        rewards_sp = sp_data["rewards"]
        arms_sp = sp_data['arms']
        agents_sp = sp_data['agents']
        sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
        
        spts_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    PendulumAgentTerminalState.__name__,
                                                    env_name,
                                                    scaling_values[np.argmax(rewards_spts_sc[:,0])])
        spts_data = load_data(spts_agent_log_dir)
        rewards_spts = spts_data["rewards"]
        arms_spts = spts_data['arms']
        agents_spts = spts_data['agents']
        spts_agent = agents_spts[np.argmax(rewards_spts[-1,:])]
        
        with plt.style.context(plt_style):
            problem_fig, problem_axs = plt.subplots(1, 2, figsize=(8,4))
            
            # Learning curve
            ax = problem_axs[0]
            problem_fig.sca(ax)
            plot_rl_exp(rewards_mp, rewards_sp, rewards_spts, rewardsRandomSc,
                        names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})",
                                f"SPAQL-TS (scaling={scaling_values[np.argmax(rewards_spts_sc[:,0])]})",
                                "Random"],
                        fig=problem_fig,
                        open_plot=False)
            ax.set_xlabel("Training iteration")
            # ax.set_ylabel("")
            # ax.set_title("Cumulative reward")
            ax.set_ylim((None, None))
            ax.xaxis.set_ticks(np.arange(0, 101, 20))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=fontsize, loc="lower right")
            
            # Number of arms
            ax = problem_axs[1]
            problem_fig.sca(ax)
            plot_rl_exp(arms_mp, arms_sp, arms_spts, 
                        names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                f"SPAQL", f"SPAQL-TS"],
                        fig=problem_fig,
                        open_plot=False,
                        add_legend=False)
            ax.set_xlabel("Training iteration")
            ax.set_ylabel("Number of arms")
            # ax.set_ylabel("")
            # ax.set_title("Number of arms")
            ax.set_yscale('log')
            ax.xaxis.set_ticks(np.arange(0, 101, 20))
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            problem_fig.tight_layout()
            problem_fig.savefig(f"{figures_dir}/{env_name}.png", dpi=300, transparent=True)
            plt.close(problem_fig)
        
        with plt.style.context(plt_style):
            ax = axs[i]
            if scale_reward:                
                ax.set_title(f"With reward scaling")
            else:
                ax.set_title(f"Without reward scaling")
            ax.xaxis.set_ticks(np.arange(0, 161, 40))
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Multi partition plots
            ax.errorbar(scaling_values, rewards_mp_sc[:,0], yerr=rewards_mp_sc[:,1]*1.96/np.sqrt(25),
                          fmt='o', color=colors[0], elinewidth=1, capsize=5, capthick=1,
                          label=f"AQL")
            ax.fill_between([scaling_values[0], scaling_values[-1]],
                            (np.average(rewards_mp_sc[:,0])+np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                            (np.average(rewards_mp_sc[:,0])-np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                            color=colors[0], alpha=0.25)
            ax.plot([scaling_values[0], scaling_values[-1]],
                    np.average(rewards_mp_sc[:,0])*np.array([1, 1]),
                    '--', color=colors[0], linewidth=1, marker=None)
            
            # Single partition plots
            ax.errorbar(scaling_values, rewards_sp_sc[:,0], yerr=rewards_sp_sc[:,1]*1.96/np.sqrt(nExp),
                          fmt='o', color=colors[1], elinewidth=1, capsize=5, capthick=1,
                          label=f"SPAQL")
            ax.fill_between([scaling_values[0], scaling_values[-1]],
                            (np.average(rewards_sp_sc[:,0])+np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                            (np.average(rewards_sp_sc[:,0])-np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                            color=colors[1], alpha=0.25)
            ax.plot([scaling_values[0], scaling_values[-1]],
                    np.average(rewards_sp_sc[:,0])*np.array([1, 1]),
                    '--', color=colors[1], linewidth=1, marker=None)
            
            ax.errorbar(scaling_values, rewards_spts_sc[:,0], yerr=rewards_spts_sc[:,1]*1.96/np.sqrt(nExp),
                          fmt='o', color=colors[2], elinewidth=1, capsize=5, capthick=1,
                          label=f"SPAQL-TS")
            ax.fill_between([scaling_values[0], scaling_values[-1]],
                            (np.average(rewards_spts_sc[:,0])+np.std(rewards_spts_sc[:,0]))*np.array([1, 1]),
                            (np.average(rewards_spts_sc[:,0])-np.std(rewards_spts_sc[:,0]))*np.array([1, 1]),
                            color=colors[2], alpha=0.25)
            ax.plot([scaling_values[0], scaling_values[-1]],
                    np.average(rewards_sp_sc[:,0])*np.array([1, 1]),
                    '--', color=colors[2], linewidth=1, marker=None)
            
            # Random plots
            ravg = np.average(rewardsRandomSc[-1,:])
            rstd = np.std(rewardsRandomSc[-1,:])
            ax.fill_between([scaling_values[0], scaling_values[-1]],
                            [ravg+rstd, ravg+rstd], [ravg-rstd, ravg-rstd],
                            color=colors[3], alpha=0.25)
            ax.plot([scaling_values[0], scaling_values[-1]],
                    [ravg, ravg],
                    '--', color=colors[3], linewidth=1, label=f"Random")
            
            ax.set_xlabel("Scaling")
            if i == 0:
                ax.set_ylabel("Cumulative reward")
            ax.legend(fontsize=fontsize, loc="lower right")
                 
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/scaling_pendulum.png", dpi=300, transparent=True)
    plt.close(fig)
    
    """
    Cartpole experiments
    """
    #
    # Set training parameters
    #
    # train_iter = 2000  # number of training iterations
    # epLen = 200        # episode length
    # nExp = 20          # number of experiments to run
    
    # n_jobs = 4
    
    with plt.style.context(plt_style):
        fig, axs = plt.subplots(1, 1, figsize=(6,4))
        # fig.suptitle("Effect of scaling parameter ")
    
    log_dir = root_log_dir + "/scaling/cartpole"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # Reward is already normalized, no need for loops
            
    # Set the environment
    env_name = "CartPole"
    env = CartPoleEnv()
    env.env.seed(seed)
    
    """
    Baselines: Random Agent
    """
    scaling = ""
    agent_class = CartPoleRandomAgent
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
    rewardsRandomSc, sc_random_agent_list = start_wrapper(agent_log_dir=agent_log_dir,
            nExp=nExp,
            agentClass=agent_class,
            classKwargs={
                'epLen': epLen,
                'numIters': None,
                },
            experimentKwargs={
                'train_iter': train_iter,
                'nExp': nExp,
                'env': env,
                'video': False,
                'debug': prints,
                'sps': False,
                },
            plot=False)
    
    rewardsRandomSc = rewardsRandomSc[:100,:]

    agent_class = MultiPartitionCartPole
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    rewards_mp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'numIters': None,
                                                    'scaling': scaling
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps': False,
                                                    },
                                                plot=False) for scaling in scaling_values)
    
    # SPAQL
    agent_class = CartPoleAgent
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
    rewards_sp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                            nExp=nExp,
                                            agentClass=agent_class,
                                            classKwargs={
                                                'epLen': epLen,
                                                'scaling': scaling,
                                                },
                                            experimentKwargs={
                                                'train_iter': train_iter,
                                                'nExp': nExp,
                                                'env': env,
                                                'video': False,
                                                'debug': prints,
                                                'sps_es': True,
                                                'u': u,
                                                'search_type': 'heuristic',
                                                'd': d,
                                                },
                                            plot=False) for scaling in scaling_values)
    
    # SPAQL-TS
    agent_class = CartPoleAgentTerminalState
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
    rewards_spts_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                            nExp=nExp,
                                            agentClass=agent_class,
                                            classKwargs={
                                                'epLen': epLen,
                                                'scaling': scaling,
                                                'lam': lam
                                                },
                                            experimentKwargs={
                                                'train_iter': train_iter,
                                                'nExp': nExp,
                                                'env': env,
                                                'video': False,
                                                'debug': prints,
                                                'sps_es': True,
                                                'u': u,
                                                'search_type': 'heuristic',
                                                'd': d,
                                                },
                                            plot=False) for scaling in scaling_values)
    
    rewards_mp_sc = np.vstack(rewards_mp_sc)
    rewards_sp_sc = np.vstack(rewards_sp_sc)
    rewards_spts_sc = np.vstack(rewards_spts_sc)
    
    opt_mp_scaling_cartpole = scaling_values[np.argmax(rewards_mp_sc[:,0])]     # 120
    opt_sp_scaling_cartpole = scaling_values[np.argmax(rewards_sp_sc[:,0])]     # 20
    opt_spts_scaling_cartpole = scaling_values[np.argmax(rewards_spts_sc[:,0])] # 20
    
    print(f"The best scaling value for the Multi Partition Agents was {scaling_values[np.argmax(rewards_mp_sc[:,0])]}")
    print(f"The best scaling value for the Single Partition Agents was {scaling_values[np.argmax(rewards_sp_sc[:,0])]}")
    print(f"The best scaling value for the Single Partition TS Agents was {scaling_values[np.argmax(rewards_spts_sc[:,0])]}")
    
    # Now that we have the best scaling values for single and multipartion,
    # it is time to plot them. We need average cumulative reward with
    # 95% confidence interval, and number of arms
    
    mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                MultiPartitionCartPole.__name__,
                                                env_name,
                                                scaling_values[np.argmax(rewards_mp_sc[:,0])])
    mp_data = load_data(mp_agent_log_dir)
    rewards_mp = mp_data["rewards"][:100,:]
    arms_mp = mp_data['arms'][:100,:]
    agents_mp = mp_data['agents']
    mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
    
    sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                CartPoleAgent.__name__,
                                                env_name,
                                                scaling_values[np.argmax(rewards_sp_sc[:,0])])
    sp_data = load_data(sp_agent_log_dir)
    rewards_sp = sp_data["rewards"][:100,:]
    arms_sp = sp_data['arms'][:100,:]
    agents_sp = sp_data['agents']
    sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
    
    spts_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                CartPoleAgentTerminalState.__name__,
                                                env_name,
                                                scaling_values[np.argmax(rewards_spts_sc[:,0])])
    spts_data = load_data(spts_agent_log_dir)
    rewards_spts = spts_data["rewards"]
    arms_spts = spts_data['arms'][:100,:]
    agents_spts = spts_data['agents']
    spts_agent = agents_spts[np.argmax(rewards_spts[-1,:])]
    
    with plt.style.context(plt_style):
        problem_fig, problem_axs = plt.subplots(1, 2, figsize=(8,4))
        
        # Learning curve
        ax = problem_axs[0]
        problem_fig.sca(ax)
        plot_rl_exp(rewards_mp, rewards_sp, rewards_spts, rewardsRandomSc,
                    names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                            f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})",
                            f"SPAQL-TS (scaling={scaling_values[np.argmax(rewards_spts_sc[:,0])]})",
                            "Random"],
                    fig=problem_fig,
                    open_plot=False)
        ax.set_xlabel("Training iteration")
        # ax.set_ylabel("")
        # ax.set_title("Cumulative reward")
        ax.set_ylim((None, None))
        # ax.xaxis.set_ticks(np.arange(0, 1001, 1000))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=fontsize, loc="lower right")
        
        # Number of arms
        ax = problem_axs[1]
        problem_fig.sca(ax)
        plot_rl_exp(arms_mp, arms_sp, arms_spts, 
                    names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                            f"SPAQL",
                            f"SPAQL-TS"],
                    fig=problem_fig,
                    open_plot=False,
                    add_legend=False)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Number of arms")
        # ax.set_ylabel("")
        # ax.set_title("Number of arms")
        # ax.xaxis.set_ticks(np.arange(0, 1001, 1000))
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        problem_fig.tight_layout()
        problem_fig.savefig(f"{figures_dir}/{env_name}.png", dpi=300, transparent=True)
        plt.close(problem_fig)
    
    with plt.style.context(plt_style):
        ax = axs
        
        # Multi partition plots
        ax.errorbar(scaling_values, rewards_mp_sc[:,0], yerr=rewards_mp_sc[:,1]*1.96/np.sqrt(25),
                      fmt='o', color=colors[0], elinewidth=1, capsize=5, capthick=1,
                      label=f"AQL")
        ax.fill_between([scaling_values[0], scaling_values[-1]],
                        (np.average(rewards_mp_sc[:,0])+np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                        (np.average(rewards_mp_sc[:,0])-np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                        color=colors[0], alpha=0.25)
        ax.plot([scaling_values[0], scaling_values[-1]],
                np.average(rewards_mp_sc[:,0])*np.array([1, 1]),
                '--', color=colors[0], linewidth=1, marker=None)
        
        # Single partition plots
        ax.errorbar(scaling_values, rewards_sp_sc[:,0], yerr=rewards_sp_sc[:,1]*1.96/np.sqrt(25),
                      fmt='o', color=colors[1], elinewidth=1, capsize=5, capthick=1,
                      label=f"SPAQL")
        ax.fill_between([scaling_values[0], scaling_values[-1]],
                        (np.average(rewards_sp_sc[:,0])+np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                        (np.average(rewards_sp_sc[:,0])-np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                        color=colors[1], alpha=0.25)
        ax.plot([scaling_values[0], scaling_values[-1]],
                np.average(rewards_sp_sc[:,0])*np.array([1, 1]),
                '--', color=colors[1], linewidth=1, marker=None)
        
        ax.errorbar(scaling_values, rewards_spts_sc[:,0], yerr=rewards_spts_sc[:,1]*1.96/np.sqrt(25),
                      fmt='o', color=colors[2], elinewidth=1, capsize=5, capthick=1,
                      label=f"SPAQL-TS")
        ax.fill_between([scaling_values[0], scaling_values[-1]],
                        (np.average(rewards_spts_sc[:,0])+np.std(rewards_spts_sc[:,0]))*np.array([1, 1]),
                        (np.average(rewards_spts_sc[:,0])-np.std(rewards_spts_sc[:,0]))*np.array([1, 1]),
                        color=colors[2], alpha=0.25)
        ax.plot([scaling_values[0], scaling_values[-1]],
                np.average(rewards_spts_sc[:,0])*np.array([1, 1]),
                '--', color=colors[2], linewidth=1, marker=None)
        
        # Random plots
        ravg = np.average(rewardsRandomSc[-1,:])
        rstd = np.std(rewardsRandomSc[-1,:])
        ax.fill_between([scaling_values[0], scaling_values[-1]],
                        [ravg+rstd, ravg+rstd], [ravg-rstd, ravg-rstd],
                        color=colors[3], alpha=0.25)
        ax.plot([scaling_values[0], scaling_values[-1]],
                [ravg, ravg],
                '--', color=colors[3], linewidth=1, label=f"Random")
        
        ax.set_xlabel("Scaling")
        ax.set_ylabel("Cumulative reward")
        ax.legend(fontsize=fontsize, loc="lower right")
               
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/scaling_cartpole.png", dpi=300, transparent=True)
    plt.close(fig)
    
    """
    Now that we have the optimal scaling parameters, we can train agents for a
    significant number of iterations.
    
    One TRPO iteration has 4000 samples. It corresponds to 20 *AQL* iterations.
    Therefore, to compare to 100 TRPO iterations, we will run 2000 *AQL* ones.
    """
    # # [20, 160]
    opt_mp_scaling_pendulum = [20.0, 160.0]
    # # [4, 10]
    opt_sp_scaling_pendulum = [4.0, 10.0]
    # # [0.4, 0]
    opt_spts_scaling_pendulum = [0.4, 0.0]
        
    opt_mp_scaling_cartpole = 120.0
    opt_sp_scaling_cartpole = 20.0
    opt_spts_scaling_cartpole = 20.0
    
    train_iter = 2000  # number of training iterations
    
    log_dir = root_log_dir + "/pendulum"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # This loop finds the best scaling value
    for i, scale_reward in enumerate([0, 1]):
        # Set the environment
        env = PendulumEnv(scale_reward=scale_reward)
        env.env.seed(seed)
            
        env_name = f"{scale_reward}_Pendulum_{train_iter}"
        
        """
        Baselines: Random Agent
        """
        agent_class = RandomAgent
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
        rewardsRandom, sc_random_agent_list = start_wrapper(agent_log_dir=agent_log_dir,
                nExp=nExp,
                agentClass=agent_class,
                classKwargs={
                    'epLen': epLen,
                    'numIters': None,
                    },
                experimentKwargs={
                    'train_iter': train_iter,
                    'nExp': nExp,
                    'env': env,
                    'video': False,
                    'debug': prints,
                    'sps': False,
                    'n_jobs': n_jobs
                    },
                plot=False)

        # AQL
        scaling = opt_mp_scaling_pendulum[i]
        agent_class = MultiPartitionPendulum
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        rewards_mp, mp_agent_list = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                    nExp=nExp,
                                                    agentClass=agent_class,
                                                    classKwargs={
                                                        'epLen': epLen,
                                                        'numIters': None,
                                                        'scaling': scaling
                                                        },
                                                    experimentKwargs={
                                                        'train_iter': train_iter,
                                                        'nExp': nExp,
                                                        'env': env,
                                                        'video': False,
                                                        'debug': prints,
                                                        'sps': False,
                                                        'n_jobs': n_jobs
                                                        },
                                                    plot=False)
        
        mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    MultiPartitionPendulum.__name__,
                                                    env_name,
                                                    scaling)
        mp_data = load_data(mp_agent_log_dir)
        rewards_mp = mp_data["rewards"]
        arms_mp = mp_data['arms']
        agents_mp = mp_data['agents']
        mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
        
        # SPAQL
        scaling = opt_sp_scaling_pendulum[i]
        agent_class = PendulumAgent
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
        rewards_sp, sp_agent_list = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'scaling': scaling,
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps_es': True,
                                                    'u': u,
                                                    'search_type': 'heuristic',
                                                    'd': d,
                                                    'n_jobs': n_jobs
                                                    },
                                                plot=False)
        
        sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    PendulumAgent.__name__,
                                                    env_name,
                                                    scaling)
        sp_data = load_data(sp_agent_log_dir)
        rewards_sp = sp_data["rewards"]
        arms_sp = sp_data['arms']
        agents_sp = sp_data['agents']
        sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
        
        # SPAQL-TS
        scaling = opt_spts_scaling_pendulum[i]
        agent_class = PendulumAgentTerminalState
        print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
        agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
        rewards_spts, spts_agent_list = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'scaling': scaling,
                                                    'lam': lam
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps_es': True,
                                                    'u': u,
                                                    'search_type': 'heuristic',
                                                    'd': d,
                                                    'n_jobs': n_jobs
                                                    },
                                                plot=False)
        
        spts_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                    PendulumAgentTerminalState.__name__,
                                                    env_name,
                                                    scaling)
        spts_data = load_data(spts_agent_log_dir)
        rewards_spts = spts_data["rewards"]
        arms_spts = spts_data['arms']
        agents_spts = spts_data['agents']
        spts_agent = agents_spts[np.argmax(rewards_spts[-1,:])]
        
        with plt.style.context(plt_style):
            problem_fig, problem_axs = plt.subplots(1, 2, figsize=(8,4))
            
            # Learning curve
            ax = problem_axs[0]
            problem_fig.sca(ax)
            plot_rl_exp(rewards_mp, rewards_sp, rewards_spts, rewardsRandom,
                        names=[f"AQL",
                                f"SPAQL",
                                f"SPAQL-TS",
                                "Random"],
                        fig=problem_fig,
                        open_plot=False)
            ax.set_xlabel("Training iteration")
            # ax.set_ylabel("")
            # ax.set_title("Cumulative reward")
            ax.set_ylim((None, None))
            ax.xaxis.set_ticks(np.arange(0, 2001, 500))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=fontsize, loc="lower right")
            
            # Number of arms
            ax = problem_axs[1]
            problem_fig.sca(ax)
            plot_rl_exp(arms_mp, arms_sp, arms_spts, 
                        names=[f"AQL",
                                f"SPAQL", f"SPAQL-TS"],
                        fig=problem_fig,
                        open_plot=False,
                        add_legend=False)
            ax.set_xlabel("Training iteration")
            ax.set_ylabel("Number of arms")
            # ax.set_ylabel("")
            # ax.set_title("Number of arms")
            ax.set_yscale('log')
            ax.xaxis.set_ticks(np.arange(0, 2001, 500))
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            problem_fig.tight_layout()
            problem_fig.savefig(f"{figures_dir}/{env_name}.png", dpi=300, transparent=True)
            plt.close(problem_fig)
            
        """
        Welch test
        """
        
        print("Applying the Welch test on the Pendulum problem...")
        
        data1 = np.abs(rewards_sp[-1,:])
        data2 = np.abs(rewards_spts[-1,:])
            
        # Significance level to be used = 0.05
        welch_test(data1, data2, 0.05, tail=2)
            
    """
    Moving to CartPole
    """
            
    log_dir = root_log_dir + "/cartpole"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # Reward is already normalized, no need for loops
            
    # Set the environment
    env_name = f"CartPole_{train_iter}"
    env = CartPoleEnv()
    env.env.seed(seed)
    
    """
    Baselines: Random Agent
    """
    scaling = ""
    agent_class = CartPoleRandomAgent
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
    rewardsRandom, sc_random_agent_list = start_wrapper(agent_log_dir=agent_log_dir,
            nExp=nExp,
            agentClass=agent_class,
            classKwargs={
                'epLen': epLen,
                'numIters': None,
                },
            experimentKwargs={
                'train_iter': train_iter,
                'nExp': nExp,
                'env': env,
                'video': False,
                'debug': prints,
                'sps': False,
                'n_jobs': n_jobs
                },
            plot=False)

    # AQL
    scaling = opt_mp_scaling_cartpole
    agent_class = MultiPartitionCartPole
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    rewards_mp_sc, _ = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                                nExp=nExp,
                                                agentClass=agent_class,
                                                classKwargs={
                                                    'epLen': epLen,
                                                    'numIters': None,
                                                    'scaling': scaling
                                                    },
                                                experimentKwargs={
                                                    'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    'debug': prints,
                                                    'sps': False,
                                                    'n_jobs': n_jobs
                                                    },
                                                plot=False)
    
    mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                MultiPartitionCartPole.__name__,
                                                env_name,
                                                scaling)
    mp_data = load_data(mp_agent_log_dir)
    rewards_mp = mp_data["rewards"]
    arms_mp = mp_data['arms']
    agents_mp = mp_data['agents']
    mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
    
    # SPAQL
    scaling = opt_sp_scaling_cartpole
    agent_class = CartPoleAgent
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
    rewards_sp_sc, _ = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                            nExp=nExp,
                                            agentClass=agent_class,
                                            classKwargs={
                                                'epLen': epLen,
                                                'scaling': scaling,
                                                },
                                            experimentKwargs={
                                                'train_iter': train_iter,
                                                'nExp': nExp,
                                                'env': env,
                                                'video': False,
                                                'debug': prints,
                                                'sps_es': True,
                                                'u': u,
                                                'search_type': 'heuristic',
                                                'd': d,
                                                'n_jobs': n_jobs
                                                },
                                            plot=False)
    
    sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                CartPoleAgent.__name__,
                                                env_name,
                                                scaling)
    sp_data = load_data(sp_agent_log_dir)
    rewards_sp = sp_data["rewards"]
    arms_sp = sp_data['arms']
    agents_sp = sp_data['agents']
    sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
    
    # SPAQL-TS
    scaling = opt_spts_scaling_cartpole
    agent_class = CartPoleAgentTerminalState
    print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
    agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, 0)
    rewards_spts_sc, _ = start_wrapper(agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
                                            nExp=nExp,
                                            agentClass=agent_class,
                                            classKwargs={
                                                'epLen': epLen,
                                                'scaling': scaling,
                                                'lam': lam
                                                },
                                            experimentKwargs={
                                                'train_iter': train_iter,
                                                'nExp': nExp,
                                                'env': env,
                                                'video': False,
                                                'debug': prints,
                                                'sps_es': True,
                                                'u': u,
                                                'search_type': 'heuristic',
                                                'd': d,
                                                'n_jobs': n_jobs
                                                },
                                            plot=False)
    
    spts_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                CartPoleAgentTerminalState.__name__,
                                                env_name,
                                                scaling)
    spts_data = load_data(spts_agent_log_dir)
    rewards_spts = spts_data["rewards"]
    arms_spts = spts_data['arms']
    agents_spts = spts_data['agents']
    spts_agent = agents_spts[np.argmax(rewards_spts[-1,:])]
    
    with plt.style.context(plt_style):
        problem_fig, problem_axs = plt.subplots(1, 2, figsize=(8,4))
        
        # Learning curve
        ax = problem_axs[0]
        problem_fig.sca(ax)
        plot_rl_exp(rewards_mp, rewards_sp, rewards_spts, rewardsRandom,
                    names=[f"AQL",
                            f"SPAQL",
                            f"SPAQL-TS",
                            "Random"],
                    fig=problem_fig,
                    open_plot=False)
        ax.set_xlabel("Training iteration")
        # ax.set_ylabel("")
        # ax.set_title("Cumulative reward")
        ax.set_ylim((None, None))
        ax.xaxis.set_ticks(np.arange(0, 2001, 500))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=fontsize, loc="lower right")
        
        # Number of arms
        ax = problem_axs[1]
        problem_fig.sca(ax)
        plot_rl_exp(arms_mp, arms_sp, arms_spts, 
                    names=[f"AQL",
                            f"SPAQL",
                            f"SPAQL-TS"],
                    fig=problem_fig,
                    open_plot=False,
                    add_legend=False)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Number of arms")
        # ax.set_ylabel("")
        # ax.set_title("Number of arms")
        ax.set_yscale('log')
        ax.xaxis.set_ticks(np.arange(0, 2001, 500))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        problem_fig.tight_layout()
        problem_fig.savefig(f"{figures_dir}/{env_name}.png", dpi=300, transparent=True)
        plt.close(problem_fig)
        
    """
    Welch test
    """
    
    print("Applying the Welch test on the CartPole problem...")
    
    data1 = np.abs(rewards_sp[-1,:])
    data2 = np.abs(rewards_spts[-1,:])
        
    # Significance level to be used = 0.05
    welch_test(data1, data2, 0.05, tail=2)
