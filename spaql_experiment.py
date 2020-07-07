#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:58:26 2020

This file implements the training logic behid AQL and SPAQL. It started as a
refactor of the Experiment class from AdaptiveQLearning

https://github.com/seanrsinclair/AdaptiveQLearning

on which most of the common code is based.

Initially, the experiments were integrated with Comet.ml. However, this
integration stopped being maintained after a while. Probably it still works.
It is necessary to set the API Key and workspace name manually. Experiments are
offline by default, and need to be uploaded manually after running.

For examples on how to run experiments, see the files in the /experiments
folder, or the code in the " if __name__=='__main__' " block at the end of this
file.

Using function start(...) to run experiments is recommended.
"""

# Necessary imports
import os
import os.path
import shutil
import pickle
import time
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from plot_rl_experiment import plot_rl_exp

# Set up Comet.ml
try:
    from comet_ml import OfflineExperiment as Experiment
except:
    Experiment = None

# Import the agents: Random, Multi Partition, Single Partition
from agents import RandomAgent, MultiPartitionAgent, \
                   SinglePartitionAgent, SinglePartitionSoftmaxAgent, \
                   SinglePartitionInfiniteAgent, SinglePartitionSoftmaxInfiniteAgent

# For the stochastic policy
from alpha_experiments import get_linear_alpha, get_exp_alpha, get_sin_alpha

# Import the envs: Oil, Ambulance
from src import environment

def store_data(file, rewards, arms, agent_list):
    """
    Store the rewards, arms, and agents to a pickled file.

    Parameters
    ----------
    file : string
        File name.
    rewards : array
        Evolution of cumulative rewards along training.
    arms : array
        Evolution of the number of arms along training.
    agent_list : list
        List with the trained agents.

    Returns
    -------
    None.

    """
    with open(file, 'wb') as f:
        pickle.dump({'rewards': rewards, 'arms': arms, 'agents': agent_list}, f)

def rollout(agent, env, debug=False, debug2=False):
    """
    Perform a rollout of an agent in an environment. Based on the code of
    AdaptiveQLearning.

    Parameters
    ----------
    agent : RandomAgent, MultiPartitionAgent, SinglePartitionAgent
        The agent to use.
    env : Oil or Ambulance
        The environment on which to perform the rollout.
    debug : Bool
        If True, print debug information.
    debug2: Bool
        If True, print additional debug information.

    Returns
    -------
    epReward : float
        Episode cumulative reward.

    """
    env.reset()
    oldState = env.state
    epReward = 0
    pContinue = 1
    h = 0
    actions = []
    while pContinue > 0 and h < env.epLen:
        # Step through the episode
        if debug: print('state : ' + str(oldState))
        action = agent.pick_action(oldState, h)
        if debug: print('action : ' + str(action))
        
        actions.append(action)

        reward, newState, pContinue = env.advance(action)
        epReward += reward
        if debug: print("reward : " + str(reward))

        oldState = newState
        h = h + 1
    
    if debug2:
        actions = list(np.around(np.array(actions),2))
        print(f"{epReward:.02f} : {actions}")
        
    return epReward

def run_aql_experiment(agent, env, nEps, nEval=20, seed=None, prints=False,
                       deBug=False, experiment=None, video=None):
    """
    The original training loop from AdaptiveQLearning, slightly modified.

    Parameters
    ----------
    agent : Agent instance
        Agent to be trained.
    env : Environemnt instance
        Environment on which the agent will be trained.
    nEps : int
        Number of training iterations (episodes).
    nEval : int, optional
        Number of evaluation rollouts at the end of each training iteration.
        The default is 20.
    seed : int, optional
        Random seed. Useful for debugging. The default is None.
    prints : bool, optional
        Print verbose information. The default is False.
    deBug : bool, optional
        If True, print debug information. The default is False.
    experiment : comet_ml.Experiment, optional
        Comet.ml Experiment instance. The default is None.
    video : bool, optional
        If True, store the information required to make a video. The default is None.

    Returns
    -------
    rewards : array
        Evolution of cumulative rewards along training.
    narms : array
        Evolution of the number of arms along training.
    agent : Agent instance
        The trained agent.

    """
    if seed: np.random.seed(seed)
    rewards = np.zeros(nEps)
    narms = np.zeros(nEps)
    if prints:
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')
    if experiment:
        experiment.log_parameters({'training_iterations': nEps,
                                   'evaluation_iterations': nEval,
                                   'seed': seed
                                   })
    for ep in range(1, nEps+1):
        if prints:
            print('Episode : ' + str(ep))
            
        # Sample
        sample_pool = []
        
        # Reset the environment
        env.reset()
        oldState = env.state
        epReward = 0

        pContinue = 1
        h = 0
        while pContinue > 0 and h < env.epLen:
            # Step through the episode
            if deBug:
                print('state : ' + str(oldState))
            action = agent.pick_action(oldState, h)
            if deBug:
                print('action : ' + str(action))

            reward, newState, pContinue = env.advance(action)
            sample_pool.append([oldState, action, reward, newState, h])
            
            # Train (online mode)
            agent.update_obs(oldState, action, reward, newState, h)
            if video:
                agent.add_frame()

            oldState = newState
            h = h + 1
        if deBug:
            print('final state: ' + str(newState))
        # print('Total Reward: ' + str(epReward))
        
        # Evaluate
        returns = []
        for n in range(nEval):
            epReward = rollout(agent, env, debug=deBug)
            returns.append(epReward)
            
        rewards[ep-1] = sum(returns)/nEval
        narms[ep-1] = agent.get_num_arms()
        
        # Log metrics to Comet.ml
        if experiment:
            experiment.log_metric("MinReturn", min(returns), epoch=ep, step=ep)
            experiment.log_metric("MaxReturn", max(returns), epoch=ep, step=ep)
            experiment.log_metric("AverageReturn", rewards[ep-1], epoch=ep, step=ep)
            experiment.log_metric("NumberArms", narms[ep-1], epoch=ep, step=ep)
            
            # TODO: Log the distribution of Q Values
    
    if prints:
        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')
        
    # Make sure memory is freed
    plt.close('all')
    
    return rewards, narms, agent # we need to return it since the original one is never edited

def run_sps_experiment(agent, env, nEps, nEval=20, seed=None, prints=False,
                       deBug=False, experiment=None, video=None, **kwargs):
    """
    SPAQL training with a fixed temperature schedule. Does not give good results.
    
     -> Use run_sps_es_experiment instead.

    Parameters
    ----------
    agent : Agent instance
        Agent to be trained.
    env : Environemnt instance
        Environment on which the agent will be trained.
    nEps : int
        Number of training iterations (episodes).
    nEval : int, optional
        Number of evaluation rollouts at the end of each training iteration.
        The default is 20.
    seed : int, optional
        Random seed. Useful for debugging. The default is None.
    prints : bool, optional
        Print verbose information. The default is False.
    deBug : bool, optional
        If True, print debug information. The default is False.
    experiment : comet_ml.Experiment, optional
        Comet.ml Experiment instance. The default is None.
    video : bool, optional
        If True, store the information required to make a video. The default is None.
    **kwargs : dict
        Additional arguments.
        
        schedule : string
            'lin' - linear decreasing schedule
            'exp' - exponential decreasing schedule
            'sin' - sinusoidal schedule
        M : float
            Maximum temperature (for the linear and exponential schedules also
            corresponds to the initial temperature).
        period : int
            Number of cycles of the sinusoidal schedule (see file alpha_experiments.py 
            for examples).

    Returns
    -------
    rewards : array
        Evolution of cumulative rewards along training.
    narms : array
        Evolution of the number of arms along training.
    agent : Agent instance
        The trained agent.

    """
    if seed: np.random.seed(seed)
    rewards = np.zeros(nEps)
    narms = np.zeros(nEps)
    if prints:
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')
    if experiment:
        experiment.log_parameters({'training_iterations': nEps,
                                   'evaluation_iterations': nEval,
                                   'seed': seed
                                   })
    # Set the temperature schedule
    assert "schedule" in kwargs
    assert "M" in kwargs
    if 'lin' in kwargs["schedule"]:
        schedule = get_linear_alpha
    elif 'exp' in kwargs["schedule"]:
        schedule = get_exp_alpha
    elif 'sin' in kwargs["schedule"]:
        schedule = get_sin_alpha
    
    # alpha = 1
    
    # Main loop
    for ep in range(1, nEps+1):
        if prints:
            print('Episode : ' + str(ep))
            
        # Sample
        sample_pool = []
        
        # Reset the environment
        env.reset()
        oldState = env.state
        epReward = 0

        if "period" in kwargs:
            alpha = schedule(c=ep, I=nEps, M=kwargs["M"], period=kwargs["period"])
        else:
            alpha = schedule(c=ep, I=nEps, M=kwargs["M"])
            
        # Uncomment the following line to force the policy to be argmax
        # alpha = 0.01

        pContinue = 1
        h = 0
        while pContinue > 0 and h < env.epLen:
            # Step through the episode
            if deBug:
                print('state : ' + str(oldState))
            action = agent.pick_action_training(oldState, h, temperature=alpha)
            if deBug:
                print('action : ' + str(action))

            reward, newState, pContinue = env.advance(action)
            sample_pool.append([oldState, action, reward, newState, h])
            
            # Train (online mode)
            agent.update_obs(oldState, action, reward, newState, h)
            if video:
                agent.add_frame()

            oldState = newState
            h = h + 1
        if deBug:
            print('final state: ' + str(newState))
        # print('Total Reward: ' + str(epReward))
        
        # Evaluate
        returns = []
        for n in range(nEval):
            epReward = rollout(agent, env, debug=deBug)
            returns.append(epReward)
            
        rewards[ep-1] = sum(returns)/nEval
        narms[ep-1] = agent.get_num_arms()
        
        # Log metrics to Comet.ml
        if experiment:
            experiment.log_metric("MinReturn", min(returns), epoch=ep, step=ep)
            experiment.log_metric("MaxReturn", max(returns), epoch=ep, step=ep)
            experiment.log_metric("AverageReturn", rewards[ep-1], epoch=ep, step=ep)
            experiment.log_metric("NumberArms", narms[ep-1], epoch=ep, step=ep)
            
            # TODO: Log the distribution of Q Values
    
    if prints:
        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')
        
    # Make sure memory is freed
    plt.close('all')
    
    return rewards, narms, agent # we need to return it since the original one is never edited

def run_sps_es_experiment(agent, env, nEps, nEval=20, seed=None, prints=False,
                       deBug=False, experiment=None, video=None, **kwargs):
    """
    Train according to the SPAQL algorithm.

    Parameters
    ----------
    agent : Agent instance
        Agent to be trained.
    env : Environemnt instance
        Environment on which the agent will be trained.
    nEps : int
        Number of training iterations (episodes).
    nEval : int, optional
        Number of evaluation rollouts at the end of each training iteration.
        The default is 20.
    seed : int, optional
        Random seed. Useful for debugging. The default is None.
    prints : bool, optional
        Print verbose information. The default is False.
    deBug : bool, optional
        If True, print debug information. The default is False.
    experiment : comet_ml.Experiment, optional
        Comet.ml Experiment instance. The default is None.
    video : bool, optional
        If True, store the information required to make a video. The default is None.
    **kwargs : dict
        Additional arguments.
        
        u : float
            Factor by which to increase the temperature
        d : float
            Factor by which to decrease u
        search_type : string
            'constrained' - the agent is reset at the end of every training
            iteration if performance did not improve
            'heuristic' - the behaviour described in the SPAQL paper
            'mixed' - A combination of both. Not implemented.
            

    Raises
    ------
    NotImplementedError
        Raised if the user tries to use the mixed search.

    Returns
    -------
    rewards : array
        Evolution of cumulative rewards along training.
    narms : array
        Evolution of the number of arms along training.
    bestAgent : Agent instance
        The best performing agent found during training.

    """
    if seed: np.random.seed(seed)
    rewards = np.zeros(nEps)
    narms = np.zeros(nEps)
    if prints:
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')
    if experiment:
        experiment.log_parameters({'training_iterations': nEps,
                                   'evaluation_iterations': nEval,
                                   'seed': seed
                                   })
    # Set the temperature schedule
    assert "u" in kwargs
    u = kwargs["u"]
    
    # Search type (default constrained)
    search_type = 'constrained' # constrained
    d = 0.5
    if 'search_type' in kwargs:
        search_type = kwargs['search_type']
    if 'd' in kwargs:
        assert d > 0 and d < 1 # Otherwise we have problems...
        d = kwargs['d']
    
    # Start from an argmax policy
    alpha = 0.01
    
    bestAgent = deepcopy(agent)
    bestReturn = sum([rollout(bestAgent, env) for i in range(nEval)])/nEval
    bestNarms = bestAgent.get_num_arms() #1
    nsplits = 0
    tempNarms = bestNarms
    
    # Main loop
    for ep in range(1, nEps+1):
        if prints:
            print('Episode : ' + str(ep))
            
        # Sample
        sample_pool = []
        
        # Reset the environment
        env.reset()
        oldState = env.state
        epReward = 0
            
        # Uncomment the following line to force the policy to be argmax
        # alpha = 0.01

        pContinue = 1
        h = 0
        while pContinue > 0 and h < env.epLen:
            # Step through the episode
            if deBug:
                print('state : ' + str(oldState))
            action = agent.pick_action_training(oldState, h, temperature=alpha)
            if deBug:
                print('action : ' + str(action))

            reward, newState, pContinue = env.advance(action)
            sample_pool.append([oldState, action, reward, newState, h])
            
            # Train (online mode)
            agent.update_obs(oldState, action, reward, newState, h)
            if video:
                agent.add_frame()

            oldState = newState
            h = h + 1
        if deBug:
            print('final state: ' + str(newState))
        # print('Total Reward: ' + str(epReward))
        
        # Evaluate
        returns = []
        for n in range(nEval):
            epReward = rollout(agent, env, debug=deBug)
            returns.append(epReward)
            
        newReturn = sum(returns)/nEval
        currReturn = newReturn
        currNarms = agent.get_num_arms() #1
        
        if tempNarms == bestNarms and currNarms != tempNarms: # First split
            tempNarms = currNarms
            # print(f"First split: {bestNarms}, {tempNarms}, {currNarms}")
        
        # If we have improvements, then we set and reset
        # Else, this means we are searching
        if newReturn >= bestReturn:
            alpha = 0.01 # Set policy to argmax again
            bestAgent = deepcopy(agent)
            bestReturn = newReturn
            bestNarms = bestAgent.get_num_arms() #1
            nsplits = 0
            if search_type == 'heuristic':
                u **= d # u = u**d. If d = 1, u does not change; if d = 0, u = 1, no change in policy
            # print(f"Episode: {ep}")
        else:
            # Always update alpha (towards a random policy)
            if alpha < 10 and tempNarms != bestNarms:
                alpha *= u
            
            # This next code snippet decides how search proceeds (constrained, free,
            # mixed, heuristically mixed, etc.)
                
            # If search is constrained then we reset the agent and that it is
            if search_type == 'constrained':
                agent = deepcopy(bestAgent) # Commenting this line means that in each
                                            # training iteration we may be departing
                                            # from a suboptimal agent. Works as sort of
                                            # divergence penalty
            elif search_type == "heuristic":
                if tempNarms != bestNarms and currNarms != tempNarms: # second split
                    # print(f"Second split: {bestNarms}, {tempNarms}, {currNarms}")
                    tempNarms = bestNarms
                    alpha = 0.01
                    agent = deepcopy(bestAgent) #1 Restart the agent
            elif search_type == "mixed":
                # Keep a count. When it reaches the end, reset
                raise NotImplementedError()
            
        rewards[ep-1] = bestReturn
        narms[ep-1] = bestAgent.get_num_arms()
        
        # Log metrics to Comet.ml
        if experiment:
            experiment.log_metric("MinReturn", min(returns), epoch=ep, step=ep)
            experiment.log_metric("MaxReturn", max(returns), epoch=ep, step=ep)
            experiment.log_metric("AverageReturn", rewards[ep-1], epoch=ep, step=ep)
            experiment.log_metric("NumberArms", narms[ep-1], epoch=ep, step=ep)
            
            # TODO: Log the distribution of Q Values
    
    if prints:
        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')
        
    # Make sure memory is freed
    plt.close('all')
    
    return rewards, narms, bestAgent # we need to return it since the original one is never edited

def run_experiment_iter(i, experiment,
                        train_iter, nExp, agent_list, env, video, user_seed,
                        experiment_name, log_params, debug,
                        project_name, sps, sps_es, **kwargs):
    """
    Function used to paralelize the run_experiment calculations.

    Parameters
    ----------
    i : int
        Index of the agent being trained.

    Raises
    ------
    NotImplementedError
        In case Comet is used, raises this error to signal where user intervention
        is required (namely to set the api_key and the workspace).

    Returns
    -------
    rewards : array
        An array with the cumulative rewards, where each column corresponds to
        an agent (random seed), and each row to a training iteration.
    arms : array
        An array with the number of agent arms, where each column corresponds
        to an agent (random seed), and each row to a training iteration.
    agent : Agent
        The trained agent.

    """
    if debug:
        start = time.time()
        print("Experiment {0} out of {1}...".format(i+1, nExp))
    if not user_seed:
        seed = int.from_bytes(os.urandom(4), 'big')
    else:
        seed = user_seed
    
    if experiment_name:
        raise NotImplementedError("Before using Comet, you need to come here and set your API key")
        experiment = Experiment(api_key=None,
                                project_name=project_name, workspace=None,
                                display_summary=False,
                                offline_directory="offline")
        experiment.add_tag(experiment_name)
        experiment.set_name("{0}_{1}".format(experiment_name, i))
        # Sometimes adding the tag fails
        log_params["experiment_tag"] = experiment_name
        experiment.log_parameters(log_params)
        
    agent = agent_list[i]
    if sps_es: # This one overrides sps
        rewards, arms, agent = run_sps_es_experiment(agent, env, train_iter,
                                                       seed=seed, video=video,
                                                       experiment=experiment,
                                                       **kwargs)
    elif sps:
        rewards, arms, agent = run_sps_experiment(agent, env, train_iter,
                                                       seed=seed, video=video,
                                                       experiment=experiment,
                                                       **kwargs)
    else:
        rewards, arms, agent = run_aql_experiment(agent, env, train_iter,
                                                       seed=seed, video=video,
                                                       experiment=experiment,
                                                       **kwargs)
    agent_list[i] = agent
    
    if experiment:
        experiment.end()
        
    if debug:
        end = time.time()
        elapsed = end - start
        units = "secs"
        if elapsed > 3600:
            elapsed /= 3600
            units = "hours"
        elif elapsed > 60:
            elapsed /= 60
            units = "mins"
        print("Time elapsed: {0:.02f} {1}".format(elapsed, units))
        
    return rewards, arms, agent

def run_experiment(train_iter, nExp, agent_list, env, video=False, user_seed=None,
                   experiment_name=None, log_params={}, debug=False,
                   project_name=None, sps=False, sps_es=False, n_jobs=1, **kwargs):
    """
    This function will train all agents in agent_list for train_iter iterations,
    nExp times.

    Parameters
    ----------
    train_iter : int
        Number of training iterations.
    nExp : int
        Number of agents to train. This is used for sanity checks.
    agent_list : list
        List of the agents which will be trained.
    env : class instance
        The environment on which the agents will be trained.
    video : bool, optional
        If True, store a video of training. The default is False.
        DEPRECATED: This consumes a lot of resources!
    user_seed : int, optional
        Random seed for the experiment. Useful when debugging. The default is
        None (seed is set automatically).
    experiment_name : str, optional
        Experiment name on Comet.ml. If passed, it will signal that you wish to
        use Comet. The default is None.
    log_params : dict, optional
        Information to be logged in Comet. The default is {}.
    debug : bool, optional
        If True, print debug information. The default is False.
    project_name : str, optional
        Project name in Comet. The default is None.
    sps : bool, optional
        Stochastic training with a pre-set temperature schedule. The default is False.
        This option is neither used or described in the companion paper.
    sps_es : bool, optional
        SPAQL training. The default is False. If both this option and the previous
        one are False, training will fall back to the original AQL algorithm.
    n_jobs : int, optional
        Number of CPU cores to use. The default is 1.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    rewards : array
        An array where each column corresponds to an agent (random seed), and
        each row to a training iteration.
    agent_list : list
        The trained agents.

    """
    assert nExp == len(agent_list)
    rewards = np.zeros([train_iter, nExp])
    arms = np.zeros([train_iter, nExp])
    experiment = None
    return_values = Parallel(n_jobs=n_jobs)(delayed(run_experiment_iter)(i, experiment, train_iter, nExp, agent_list, env, video, user_seed, experiment_name, log_params, debug, project_name, sps, sps_es,
                                                         **kwargs) for i in range(nExp))
    for i, content in enumerate(return_values):
        rewards[:,i] = content[0]
        arms[:,i] = content[1]
        agent_list[i] = content[2]
        
    return rewards, arms
    
    # Below this line is the original non-parallel code for run_experiment.
    
    # for i in range(nExp):
    #     if debug:
    #         start = time.time()
    #         print("Experiment {0} out of {1}...".format(i+1, nExp))
    #     if not user_seed:
    #         seed = int.from_bytes(os.urandom(4), 'big')
    #     else:
    #         seed = user_seed
        
    #     if experiment_name:
    #         experiment = Experiment(api_key="i1YETHvuaTOXQI3CC98Yf17dH",
    #                                 project_name=project_name, workspace="jaraujo98",
    #                                 display_summary=False,
    #                                 offline_directory="offline")
    #         experiment.add_tag(experiment_name)
    #         experiment.set_name("{0}_{1}".format(experiment_name, i))
    #         # Sometimes adding the tag fails
    #         log_params["experiment_tag"] = experiment_name
    #         experiment.log_parameters(log_params)
            
    #     agent = agent_list[i]
    #     if sps_es: # This one overrides sps
    #         rewards[:, i], arms[:, i], agent = run_sps_es_experiment(agent, env,
    #                                                                  train_iter,
    #                                                        seed=seed, video=video,
    #                                                        experiment=experiment,
    #                                                        **kwargs)
    #         agent_list[i] = agent
    #     elif sps:
    #         rewards[:, i], arms[:, i] = run_sps_experiment(agent, env, train_iter,
    #                                                        seed=seed, video=video,
    #                                                        experiment=experiment,
    #                                                        **kwargs)
    #     else:
    #         rewards[:, i], arms[:, i] = run_aql_experiment(agent, env, train_iter,
    #                                                        seed=seed, video=video,
    #                                                        experiment=experiment,
    #                                                        **kwargs)
    #     if experiment:
    #         experiment.end()
            
    #     if debug:
    #         end = time.time()
    #         elapsed = end - start
    #         units = "secs"
    #         if elapsed > 3600:
    #             elapsed /= 3600
    #             units = "hours"
    #         elif elapsed > 60:
    #             elapsed /= 60
    #             units = "mins"
    #         print("Time elapsed: {0:.02f} {1}".format(elapsed, units))

def start(agent_log_dir, nExp, agentClass, classKwargs, experimentKwargs,
          plot=False):
    """
    Train a batch of agents.

    Parameters
    ----------
    agent_log_dir : string
        Directory where results will be stored.
    nExp : int
        Number of agents to train.
    agentClass : Agent class
        Class from which the agents will be instantiated.
    classKwargs : dict
        Input arguments for initializing the Agent class instance
    experimentKwargs : dict
        Input arguments for the experiment function.
    plot : bool, optional
        At the end of training, open a figure with a plot of the learning curve.
        The default is False.

    Returns
    -------
    rewards : array
        An array where each column corresponds to an agent (random seed), and
        each row to a training iteration.
    agent_list : list
        The trained agents.

    """
    if not os.path.exists(agent_log_dir):
        os.mkdir(agent_log_dir)
    agent_list = []
    for i in range(nExp):
        agent_list.append(agentClass(**classKwargs))
    rewards, arms = run_experiment(agent_list=agent_list, **experimentKwargs)
    store_data("{0}/agentData.pk".format(agent_log_dir), rewards, arms, agent_list)
    
    # Store the videos
    if 'video' in experimentKwargs:
        store_videos = experimentKwargs['video']
        for i, agent in enumerate(agent_list):
            video_name = "{0}/agent_{1}.mp4".format(agent_log_dir,i)
            if store_videos:
                agent.export_video(name=video_name)
    
    if plot: plot_rl_exp(rewards)
    
    return rewards, agent_list

if __name__ == "__main__":
    #
    # Set parameters
    #
    train_iter = 10  # number of training iterations
    epLen = 5          # episode length
    scaling = 0.5      # scaling factor for the bonus
    nExp = 20          # number of experiments to run
    
    store_videos = False # Export training videos
                         # DEPRECATED: Exporting videos is highly time consuming
                         # see https://github.com/matplotlib/matplotlib/issues/10207
    log_dir = "SPAQLOilResults" # Directory where everything will be logged
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    # Random seed
    seed = None
    
    # Set the environments
    lam = 1
    starting_state = 0
    env = environment.makeLaplaceOil(epLen, lam, starting_state)

    #
    # Gather baseline information
    #
    
    # Random agent
    print("Training the random agents...")
    r_log_dir = "{}/RandomResults".format(log_dir)
    if not os.path.exists(r_log_dir):
        os.mkdir(r_log_dir)
    r_agent_list = []
    for i in range(nExp):
        r_agent_list.append(RandomAgent(epLen, None))
    rRewards, rArms = run_experiment(train_iter, nExp, r_agent_list, env,
                                      # experiment_name="RandomAgent",
                                      # log_params={"log_dir": r_log_dir},
                                      debug=True)
    store_data("{0}/rAgentData.pk".format(r_log_dir), rRewards, rArms, r_agent_list)
    
    # Multi partition agent
    print("Training AQL agents...")
    mp_log_dir = "{}/MultiPartitionResults".format(log_dir)
    if not os.path.exists(mp_log_dir):
        os.mkdir(mp_log_dir)
    mp_agent_list = []
    for i in range(nExp):
        mp_agent_list.append(MultiPartitionAgent(epLen, None, scaling))
    mpRewards, mpArms = run_experiment(train_iter, nExp, mp_agent_list, env,
                                        video=False,
                                        # experiment_name="MultiPartitionAgent",
                                        # log_params={"log_dir": mp_log_dir},
                                        debug=True)
    store_data("{0}/mpAgentData.pk".format(mp_log_dir), mpRewards, mpArms, mp_agent_list)
    
    # # Store the videos
    # for i, agent in enumerate(mp_agent_list):
    #     video_name = "{0}/MultiPartitionAgent_{1}.mp4".format(mp_log_dir,i)
    #     video_folder = video_name.split(".")[0]
    #     if store_videos and os.path.exists(video_folder):
    #         shutil.rmtree(video_folder)
        
    #     if store_videos:
    #         agent.export_video(name=video_name)
    
    # Single partition agent
    print("Training SPAQL agents...")
    sp_log_dir = "{}/SinglePartitionResults".format(log_dir)
    if not os.path.exists(sp_log_dir):
        os.mkdir(sp_log_dir)
    sp_agent_list = []
    for i in range(nExp):
        sp_agent_list.append(SinglePartitionAgent(epLen, None, scaling))
    spRewards, spArms = run_experiment(train_iter, nExp, sp_agent_list, env,
                                        video=False,
                                        # experiment_name="SinglePartitionAgent",
                                        # log_params={"log_dir": sp_log_dir},
                                        debug=True)
    store_data("{0}/spAgentData.pk".format(sp_log_dir), spRewards, spArms, sp_agent_list)
    
    # # Store the videos
    # for i, agent in enumerate(sp_agent_list):
    #     video_name = "{0}/SinglePartitionAgent_{1}.mp4".format(sp_log_dir,i)
    #     if store_videos:
    #         agent.export_video(name=video_name)

    # Single partition infinite agent
    print("Training SPAQL (1) agents...")
    agent_log_dir = "{}/SinglePartitionInfiniteResults".format(log_dir)
    rewardsDet, spi_agent_list = start(agent_log_dir=agent_log_dir,
                                  nExp=nExp,
                                  agentClass=SinglePartitionInfiniteAgent,
                                  classKwargs={'epLen': epLen,
                                                'numIters': None,
                                                'scaling': scaling},
                                  experimentKwargs={'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    #'experiment_name': "SinglePartitionInfiniteAgent",
                                                    #'project_name': "spiaql-oil",
                                                    #'log_params': {"log_dir": agent_log_dir},
                                                    'debug': True,
                                                    #'seed': 1,
                                                    #'deBug': 1,
                                                    #'nEval': 1
                                                    },
                                  plot=False
                                  )
    
    # Single partition infinite softmax agent
    print("Training SPAQL (2) agents...")
    agent_log_dir = "{}/SinglePartitionSoftmaxInfiniteResults".format(log_dir)
    rewardsSof, spsi_agent_list = start(agent_log_dir=agent_log_dir,
                                  nExp=nExp,
                                  agentClass=SinglePartitionSoftmaxInfiniteAgent,
                                  classKwargs={'epLen': epLen,
                                                'numIters': None,
                                                'scaling': scaling},
                                  experimentKwargs={'train_iter': train_iter,
                                                    'nExp': nExp,
                                                    'env': env,
                                                    'video': False,
                                                    #'experiment_name': "SinglePartitionSoftmaxAgent",
                                                    #'project_name': "spiaql-oil",
                                                    'log_params': {"log_dir": agent_log_dir},
                                                    'debug': True,
                                                    'sps': True,
                                                    'schedule': 'exp',
                                                    'M': 10,
                                                    #'seed': 1,
                                                    #'deBug': 1,
                                                    #'nEval': 1
                                                    },
                                  plot=False
                                  )
    
    # Single partition finite softmax agent
    print("Training SPAQL (3) agents...")
    agent_log_dir = "{}/SinglePartitionSoftmaxResults".format(log_dir)
    rewardsFinSof, spsf_agent_list = start(agent_log_dir=agent_log_dir,
                                      nExp=nExp,
                                      agentClass=SinglePartitionSoftmaxAgent,
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
                                          #'experiment_name': "SinglePartitionSoftmaxAgent",
                                          #'project_name': "spiaql-oil",
                                          'log_params': {"log_dir": agent_log_dir},
                                          'debug': True,
                                          'sps': True,
                                          'schedule': 'exp',
                                          'M': 10,
                                          #'seed': 1,
                                          #'deBug': 1,
                                          #'nEval': 1
                                          },
                                      plot=False
                                      )
    
    plot_rl_exp(rewardsDet, rewardsSof, rewardsFinSof, names=["Argmax", "Softmax", "Softmax Finite"])
    
    # plt.close('all')
    # plot_rl_exp(rRewards, mpRewards, spRewards,
    #             names=["Random", "MultiPartition", "SinglePartition"],
    #             plot_file_name="spaql_oil.png")
