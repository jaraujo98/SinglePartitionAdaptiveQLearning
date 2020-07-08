#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:24:51 2020

This file generates the images for the single partition adaptive Q-learning
paper.
"""

import os
import sys
import pickle
from copy import deepcopy
import time
import numpy as np
import matplotlib.pyplot as plt

plt_style = "fivethirtyeight"
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# The code in this file is highly paralelizable
from joblib import Parallel, delayed
from functools import partial
n_jobs = 4

# Import the agents: Random, Multi Partition, Single Partition
from agents import RandomAgent, MultiPartitionAgent, \
                   SinglePartitionSoftmaxInfiniteAgent

# Import the envs: Oil, Ambulance
from envs import OilLaplace, OilQuadratic, Ambulance, beta, uniform

# Experimental setup
from spaql_experiment import start

from plot_rl_experiment import plot_rl_exp

def load_data(log_dir):
    """
    Reads the agentData.pk file in folder log_dir.

    Parameters
    ----------
    log_dir : str
        The directory where agentData.pk is stored.

    Returns
    -------
    None.

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
    input("""WARNING: Running this file as it is will duplicate the experiments in the SPAQL paper.
These experiments take a long time to run, and the results occupy some GB.

Press Enter to continue...""")
    #
    # Set training parameters
    #
    train_iter = 5000  # number of training iterations
    epLen = 5          # episode length
    nExp = 25          # number of experiments to run
    
    n_jobs = 4
    
    #
    # Set algorithm parameters
    #
    M = 0.1
    u = 2
    d = 0.8
    
    """
    Oil experiments
    """
    if not os.path.exists("results"):
        os.mkdir("results")
    
    root_log_dir = "results/SPAQLPaperResults" # Directory where everything will be logged
    figures_dir = root_log_dir + "/figures"
    scaling_dir = root_log_dir + "/scaling"
    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
        os.mkdir(figures_dir)
        os.mkdir(scaling_dir)
    
    prints = False
    
    # Random seed
    seed = None
    
    scaling_values = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5]
    
    with plt.style.context(plt_style):
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12,8))
        # fig.suptitle("Effect of scaling parameter ")
        axs[0,0].set_ylim((-0.1, 5.1))
        axs[1,0].set_ylim((-0.1, 5.1))
    
    log_dir = root_log_dir + "/scaling/oil"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # This loop finds the best scaling value
    for j, lam in enumerate([1, 10, 50]):
        for i, name in enumerate(["OilLaplace", "OilQuadratic"]):
            
            # Set the environment
            starting_state = 0
            if name == "OilLaplace":
                env = OilLaplace(epLen, starting_state, lam)
            else:
                env = OilQuadratic(epLen, starting_state, lam)
                
            env_name = f"{lam}_{name}"
            
            """
            Baselines: Random Agent
            """
            scaling = ""
            agent_class = RandomAgent
            print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
            agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
            try:
                with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
                    data = pickle.load(f)
                    rewardsRandomSc = data["rewards"]
            except:
                now = time.time()
                rewardsRandomSc, sc_random_agent_list = start(agent_log_dir=agent_log_dir,
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
                                                            #'seed': 1,
                                                            #'deBug': 1,
                                                            #'nEval': 1
                                                            },
                                                        plot=False)
                print(f"Time elapsed: {time.time() - now}")

            agent_class = MultiPartitionAgent
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
                                                            #'seed': 1,
                                                            #'deBug': 1,
                                                            #'nEval': 1
                                                            },
                                                        plot=False) for scaling in scaling_values)
            
            agent_class = SinglePartitionSoftmaxInfiniteAgent
            print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
            agent_log_dir = "{0}/{1}{2}{3}".format(log_dir, agent_class.__name__, env_name, scaling)
            rewards_sp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
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
                                                        'sps_es': True,
                                                        'u': u,
                                                        'search_type': 'heuristic',
                                                        'd': d,
                                                        #'seed': 1,
                                                        #'deBug': 1,
                                                        #'nEval': 1,
                                                        #'user_seed': seed
                                                        },
                                                    plot=False) for scaling in scaling_values)
            
            rewards_mp_sc = np.vstack(rewards_mp_sc)
            rewards_sp_sc = np.vstack(rewards_sp_sc)
            
            print(f"The best scaling value for the Multi Partition Agents was {scaling_values[np.argmax(rewards_mp_sc[:,0])]}")
            print(f"The best scaling value for the Single Partition Agents was {scaling_values[np.argmax(rewards_sp_sc[:,0])]}\n")
            
            # Now that we have the best scaling values for single and multipartion,
            # it is time to plot them. We need average cumulative reward with
            # 95% confidence interval, number of arms, and a plot of the single
            # partition
            
            mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                        MultiPartitionAgent.__name__,
                                                        env_name,
                                                        scaling_values[np.argmax(rewards_mp_sc[:,0])])
            mp_data = load_data(mp_agent_log_dir)
            rewards_mp = mp_data["rewards"]
            arms_mp = mp_data['arms']
            agents_mp = mp_data['agents']
            mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
            
            sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                        SinglePartitionSoftmaxInfiniteAgent.__name__,
                                                        env_name,
                                                        scaling_values[np.argmax(rewards_sp_sc[:,0])])
            sp_data = load_data(sp_agent_log_dir)
            rewards_sp = sp_data["rewards"]
            arms_sp = sp_data['arms']
            agents_sp = sp_data['agents']
            sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
            
            with plt.style.context(plt_style):
                problem_fig, problem_axs = plt.subplots(2, 3, figsize=(12,8))
                
                # Learning curve
                ax = problem_axs[0,0]
                problem_fig.sca(ax)
                plot_rl_exp(rewards_mp, rewards_sp, rewardsRandomSc,
                            names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                    f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})",
                                    "Random"],
                            fig=problem_fig,
                            open_plot=False)
                ax.set_xlabel("Training iteration")
                # ax.set_ylabel("")
                # ax.set_title("Cumulative reward")
                ax.set_ylim((-0.1, 5.1))
                ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Number of arms
                ax = problem_axs[0,1]
                problem_fig.sca(ax)
                plot_rl_exp(arms_mp, arms_sp, 
                            names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                    f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})"],
                            fig=problem_fig,
                            open_plot=False,
                            add_legend=False)
                ax.set_xlabel("Training iteration")
                ax.set_ylabel("Number of arms")
                # ax.set_ylabel("")
                # ax.set_title("Number of arms")
                ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Single Partition
                ax = problem_axs[0,2]
                problem_fig.sca(ax)
                sp_agent.tree.plot(0)
                ax.set_title("Single Partition")
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Three time steps for the Multi Partition
                for k in range(3):
                    ax = problem_axs[1,k]
                    ax.set_title(f"Multi Partition (t={2*k+1})")
                    problem_fig.sca(ax)
                    mp_agent.tree_list[2*k].plot(0)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                
                problem_fig.tight_layout()
                problem_fig.savefig(f"{figures_dir}/{name}{lam}.png", dpi=300, transparent=True)
                plt.close(problem_fig)
            
            with plt.style.context(plt_style):
                ax = axs[i,j]
                ax.set_title(f"{name.replace('Oil', 'Oil ')} ($\lambda$={lam})")
                
                # Multi partition plots
                ax.errorbar(scaling_values, rewards_mp_sc[:,0], yerr=rewards_mp_sc[:,1]*1.96/np.sqrt(25),
                              fmt='o', elinewidth=1, capsize=5, capthick=1,
                              label=f"AQL")
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                (np.average(rewards_mp_sc[:,0])+np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                                (np.average(rewards_mp_sc[:,0])-np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                                color="#008fd5", alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        np.average(rewards_mp_sc[:,0])*np.array([1, 1]),
                        '--', color="#008fd5", linewidth=1, marker=None)
                
                # Single partition plots
                ax.errorbar(scaling_values, rewards_sp_sc[:,0], yerr=rewards_sp_sc[:,1]*1.96/np.sqrt(25),
                              fmt='o', elinewidth=1, capsize=5, capthick=1,
                              label=f"SPAQL")
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                (np.average(rewards_sp_sc[:,0])+np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                                (np.average(rewards_sp_sc[:,0])-np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                                color="#fc4f30", alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        np.average(rewards_sp_sc[:,0])*np.array([1, 1]),
                        '--', color="#fc4f30", linewidth=1, marker=None)
                
                # Random plots
                ravg = np.average(rewardsRandomSc[-1,:])
                rstd = np.std(rewardsRandomSc[-1,:])
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                [ravg+rstd, ravg+rstd], [ravg-rstd, ravg-rstd],
                                color='green', alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        [ravg, ravg],
                        'g--', linewidth=1, label=f"Random")
                
                # Maximum reward
                ravg = 4.25
                rstd = 0
                ax.plot([scaling_values[0], scaling_values[-1]],
                        [ravg, ravg],
                        '--', linewidth=2, label=f"Maximum")
                
                ax.legend(loc='lower left')
                if i == 1:
                    ax.set_xlabel(r"Scaling ($\xi$)")
                if i == 0 and j == 2:
                    ax.legend(loc='upper right')
                if j == 0:
                    ax.set_ylabel("Cumulative reward")
            
            if name == "OilLaplace" and lam == 1:
                """
                Knowing the best scaling values, perform the OilLaplace experiment
                with an episode length of 50.
                
                Delete the if statement and dedent this entire block if you would
                like to perform this experiment for all Oil scenarios.
                
                The number of experiments was manually overriden with 2 in the
                experimentKwargs variable. Edit at will.
                """
                # Set the environment
                newEpLen = 50
                starting_state = 0
                if name == "OilLaplace":
                    env = OilLaplace(newEpLen, starting_state, lam)
                else:
                    env = OilQuadratic(newEpLen, starting_state, lam)
                
                best_mp_scaling = scaling_values[np.argmax(rewards_mp_sc[:,0])]
                best_sp_scaling = scaling_values[np.argmax(rewards_sp_sc[:,0])]
                
                # Random Baseline
                scaling = ""
                agent_class = RandomAgent
                print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
                agent_log_dir = "{0}/Random_epLen{1}".format(log_dir, newEpLen)
                try:
                    with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
                        data = pickle.load(f)
                        rewardsRandom50 = data["rewards"]
                except:
                    now = time.time()
                    rewardsRandom50, random_agent_list_50 = start(agent_log_dir=agent_log_dir,
                                                            nExp=nExp,
                                                            agentClass=agent_class,
                                                            classKwargs={
                                                                'epLen': newEpLen,
                                                                'numIters': None,
                                                                },
                                                            experimentKwargs={
                                                                'train_iter': train_iter,
                                                                'nExp': nExp,
                                                                'env': env,
                                                                'video': False,
                                                                'debug': prints,
                                                                'sps': False,
                                                                #'seed': 1,
                                                                #'deBug': 1,
                                                                #'nEval': 1
                                                                },
                                                            plot=False)
                    print(f"Time elapsed: {time.time() - now}")
                
                # Train AQL agent
                agent_class = MultiPartitionAgent
                agent_log_dir="{0}/MultiPartition_epLen{1}".format(log_dir, newEpLen)
                print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem with 50 steps...")
                try:
                    with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
                        data = pickle.load(f)
                        rewardsMulti50 = data["rewards"]
                except:
                    now = time.time()
                    rewardsMulti50, multi_agent_list_50 = start(agent_log_dir=agent_log_dir,
                                                            nExp=2,#nExp,
                                                            agentClass=agent_class,
                                                            classKwargs={
                                                                'epLen': newEpLen,
                                                                'numIters': None,
                                                                'scaling': best_mp_scaling
                                                                },
                                                            experimentKwargs={
                                                                'train_iter': train_iter,
                                                                'nExp': 2,#nExp,
                                                                'env': env,
                                                                'video': False,
                                                                'debug': prints,
                                                                'sps': False,
                                                                #'seed': 1,
                                                                #'deBug': 1,
                                                                #'nEval': 1
                                                                },
                                                            plot=False)
                    print(f"Time elapsed: {time.time() - now}")
                
                mp_data = load_data(agent_log_dir)
                rewards_mp = mp_data["rewards"]
                arms_mp = mp_data['arms']
                agents_mp = mp_data['agents']
                mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
                
                # Train SPAQL agent
                agent_class = SinglePartitionSoftmaxInfiniteAgent
                agent_log_dir="{0}/SinglePartition_epLen{1}".format(log_dir, newEpLen)
                print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem with 50 steps...")
                try:
                    with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
                        data = pickle.load(f)
                        rewardsSingle50 = data["rewards"]
                except:
                    now = time.time()
                    rewardsSingle50, single_agent_list_50 = start(agent_log_dir=agent_log_dir,
                                                            nExp=2,#nExp,
                                                            agentClass=agent_class,
                                                            classKwargs={
                                                                'epLen': newEpLen,
                                                                'numIters': None,
                                                                'scaling': best_sp_scaling
                                                                },
                                                            experimentKwargs={
                                                                'train_iter': train_iter,
                                                                'nExp': 2,#nExp,
                                                                'env': env,
                                                                'video': False,
                                                                'debug': prints,
                                                                'sps_es': True,
                                                                'u': u,
                                                                'search_type': 'heuristic',
                                                                'd': d,
                                                                #'seed': 1,
                                                                #'deBug': 1,
                                                                #'nEval': 1
                                                                },
                                                            plot=False)
                    print(f"Time elapsed: {time.time() - now}")
                
                sp_data = load_data(agent_log_dir)
                rewards_sp = sp_data["rewards"]
                arms_sp = sp_data['arms']
                agents_sp = sp_data['agents']
                sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
                
                with plt.style.context(plt_style):
                    problem_fig, problem_axs = plt.subplots(2, 3, figsize=(12,8))
                    
                    # Learning curve
                    ax = problem_axs[0,0]
                    problem_fig.sca(ax)
                    plot_rl_exp(rewardsMulti50, rewardsSingle50, rewardsRandom50,
                                names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                        f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})",
                                        "Random"],
                                fig=problem_fig,
                                open_plot=False)
                    ax.set_xlabel("Training iteration")
                    # ax.set_ylabel("")
                    # ax.set_title("Cumulative reward")
                    ax.set_ylim((-0.1, newEpLen + 0.1))
                    ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    
                    # Number of arms
                    ax = problem_axs[0,1]
                    problem_fig.sca(ax)
                    plot_rl_exp(arms_mp, arms_sp, 
                                names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                        f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})"],
                                fig=problem_fig,
                                open_plot=False,
                                add_legend=False)
                    ax.set_xlabel("Training iteration")
                    ax.set_ylabel("Number of arms")
                    # ax.set_ylabel("")
                    # ax.set_title("Number of arms")
                    ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    
                    # Single Partition
                    ax = problem_axs[0,2]
                    problem_fig.sca(ax)
                    sp_agent.tree.plot(0)
                    ax.set_title("Single Partition")
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    
                    # Three time steps for the Multi Partition
                    for k in range(3):
                        t = int(np.floor(1+24.5*k))
                        ax = problem_axs[1,k]
                        ax.set_title(f"Multi Partition (t={t})")
                        problem_fig.sca(ax)
                        mp_agent.tree_list[t-1].plot(0)
                        ax.tick_params(axis='both', which='major', labelsize=10)
                    
                    problem_fig.tight_layout()
                    problem_fig.savefig(f"{figures_dir}/{name}{lam}_epLen{newEpLen}.png", dpi=300, transparent=True)
                    plt.close(problem_fig)
            
                    
    fig.savefig(f"{figures_dir}/scaling_oil.png", dpi=300, transparent=True)
    plt.close(fig)
    
    """
    Ambulance experiments
    """
    #
    # Set training parameters
    #
    train_iter = 2000  # number of training iterations
    epLen = 5          # episode length
    nExp = 50          # number of experiments to run
    
    n_jobs = 4
    
    with plt.style.context(plt_style):
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12,8))
        # fig.suptitle("Effect of scaling parameter ")
        axs[0,0].set_ylim((-0.1, 5.1))
        axs[1,0].set_ylim((-0.1, 5.1))
    
    log_dir = root_log_dir + "/scaling/ambulance"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # This loop finds the best scaling value
    for j, c in enumerate([0, 0.25, 1]):
        for i, name in enumerate(["Beta", "Uniform"]):
            
            # Set the environment
            starting_state = 0.5
            if name == "Beta":
                env = Ambulance(epLen, beta, c, starting_state)
            else:
                env = Ambulance(epLen, uniform, c, starting_state)
                
            env_name = f"{c}_{name}"
            
            """
            Baselines: Random Agent
            """
            scaling = ""
            agent_class = RandomAgent
            print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
            agent_log_dir = "{0}/{1}_{2}".format(log_dir, agent_class.__name__, env_name)
            try:
                with open(f"{agent_log_dir}/agentData.pk", 'rb') as f:
                    data = pickle.load(f)
                    rewardsRandomSc = data["rewards"]
            except:
                now = time.time()
                rewardsRandomSc, sc_random_agent_list = start(agent_log_dir=agent_log_dir,
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
                                                            #'seed': 1,
                                                            #'deBug': 1,
                                                            #'nEval': 1
                                                            },
                                                        plot=False)
                print(f"Time elapsed: {time.time() - now}")

            agent_class = MultiPartitionAgent
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
                                                            #'seed': 1,
                                                            #'deBug': 1,
                                                            #'nEval': 1
                                                            },
                                                        plot=False) for scaling in scaling_values)
            
            agent_class = SinglePartitionSoftmaxInfiniteAgent
            print(f"Training {nExp} {agent_class.__name__} on the {env_name} problem...")
            agent_log_dir = "{0}/{1}{2}{3}".format(log_dir, agent_class.__name__, env_name, scaling)
            rewards_sp_sc = Parallel(n_jobs=n_jobs) (delayed(scaling_wrapper) (agent_log_dir="{0}/{1}_{2}_{3}".format(log_dir, agent_class.__name__, env_name, scaling),
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
                                                        'sps_es': True,
                                                        'u': u,
                                                        'search_type': 'heuristic',
                                                        'd': d,
                                                        #'seed': 1,
                                                        #'deBug': 1,
                                                        #'nEval': 1,
                                                        #'user_seed': seed
                                                        },
                                                    plot=False) for scaling in scaling_values)
            
            rewards_mp_sc = np.vstack(rewards_mp_sc)
            rewards_sp_sc = np.vstack(rewards_sp_sc)
            
            print(f"The best scaling value for the Multi Partition Agents was {scaling_values[np.argmax(rewards_mp_sc[:,0])]}")
            print(f"The best scaling value for the Single Partition Agents was {scaling_values[np.argmax(rewards_sp_sc[:,0])]}\n")
            
            # Now that we have the best scaling values for single and multipartion,
            # it is time to plot them. We need average cumulative reward with
            # 95% confidence interval, number of arms, and a plot of the single
            # partition
            
            mp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                        MultiPartitionAgent.__name__,
                                                        env_name,
                                                        scaling_values[np.argmax(rewards_mp_sc[:,0])])
            mp_data = load_data(mp_agent_log_dir)
            rewards_mp = mp_data["rewards"]
            arms_mp = mp_data['arms']
            agents_mp = mp_data['agents']
            mp_agent = agents_mp[np.argmax(rewards_mp[-1,:])]
            
            sp_agent_log_dir = "{0}/{1}_{2}_{3}".format(log_dir,
                                                        SinglePartitionSoftmaxInfiniteAgent.__name__,
                                                        env_name,
                                                        scaling_values[np.argmax(rewards_sp_sc[:,0])])
            sp_data = load_data(sp_agent_log_dir)
            rewards_sp = sp_data["rewards"]
            arms_sp = sp_data['arms']
            agents_sp = sp_data['agents']
            sp_agent = agents_sp[np.argmax(rewards_sp[-1,:])]
            
            with plt.style.context(plt_style):
                problem_fig, problem_axs = plt.subplots(2, 3, figsize=(12,8))
                
                # Learning curve
                ax = problem_axs[0,0]
                problem_fig.sca(ax)
                plot_rl_exp(rewards_mp, rewards_sp, rewardsRandomSc,
                            names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                   f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})",
                                   "Random"],
                            fig=problem_fig,
                            open_plot=False)
                ax.set_xlabel("Training iteration")
                # ax.set_ylabel("")
                # ax.set_title("Cumulative reward")
                ax.set_ylim((-0.1, 5.1))
                ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Number of arms
                ax = problem_axs[0,1]
                problem_fig.sca(ax)
                plot_rl_exp(arms_mp, arms_sp, 
                            names=[f"AQL (scaling={scaling_values[np.argmax(rewards_mp_sc[:,0])]})",
                                   f"SPAQL (scaling={scaling_values[np.argmax(rewards_sp_sc[:,0])]})"],
                            fig=problem_fig,
                            open_plot=False,
                            add_legend=False)
                ax.set_xlabel("Training iteration")
                ax.set_ylabel("Number of arms")
                # ax.set_ylabel("")
                # ax.set_title("Number of arms")
                ax.xaxis.set_ticks(np.arange(0, train_iter + 1, train_iter // 5))
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Single Partition
                ax = problem_axs[0,2]
                problem_fig.sca(ax)
                sp_agent.tree.plot(0)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.set_title("Single Partition")
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Three time steps for the Multi Partition
                for k in range(3):
                    ax = problem_axs[1,k]
                    ax.set_title(f"Multi Partition (t={2*k+1})")
                    problem_fig.sca(ax)
                    mp_agent.tree_list[2*k].plot(0)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                
                problem_fig.tight_layout()
                problem_fig.savefig(f"{figures_dir}/{name}{c}.png", dpi=300, transparent=True)
                plt.close(problem_fig)
            
            with plt.style.context(plt_style):
                ax = axs[i,j]
                ax.set_title(f"{name} (c={c})")
                
                # Multi partition plots
                ax.errorbar(scaling_values, rewards_mp_sc[:,0], yerr=rewards_mp_sc[:,1]*1.96/np.sqrt(50),
                             fmt='o', elinewidth=1, capsize=5, capthick=1,
                             label=f"AQL")
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                (np.average(rewards_mp_sc[:,0])+np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                                (np.average(rewards_mp_sc[:,0])-np.std(rewards_mp_sc[:,0]))*np.array([1, 1]),
                                color="#008fd5", alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        np.average(rewards_mp_sc[:,0])*np.array([1, 1]),
                        '--', color="#008fd5", linewidth=1, marker=None)
                
                # Single partition plots
                ax.errorbar(scaling_values, rewards_sp_sc[:,0], yerr=rewards_sp_sc[:,1]*1.96/np.sqrt(50),
                             fmt='o', elinewidth=1, capsize=5, capthick=1,
                             label=f"SPAQL")
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                (np.average(rewards_sp_sc[:,0])+np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                                (np.average(rewards_sp_sc[:,0])-np.std(rewards_sp_sc[:,0]))*np.array([1, 1]),
                                color="#fc4f30", alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        np.average(rewards_sp_sc[:,0])*np.array([1, 1]),
                        '--', color="#fc4f30", linewidth=1, marker=None)
                
                # Random plots
                ravg = np.average(rewardsRandomSc[-1,:])
                rstd = np.std(rewardsRandomSc[-1,:])
                ax.fill_between([scaling_values[0], scaling_values[-1]],
                                [ravg+rstd, ravg+rstd], [ravg-rstd, ravg-rstd],
                                color='green', alpha=0.25)
                ax.plot([scaling_values[0], scaling_values[-1]],
                        [ravg, ravg],
                        'g--', linewidth=1, label=f"Random")
                
                ax.legend(loc='lower left')
                if i == 1:
                    ax.set_xlabel(r"Scaling ($\xi$)")
                if j == 0:
                    ax.set_ylabel("Cumulative reward")
                    
    fig.savefig(f"{figures_dir}/scaling_ambulance.png", dpi=300, transparent=True)
    plt.close(fig)