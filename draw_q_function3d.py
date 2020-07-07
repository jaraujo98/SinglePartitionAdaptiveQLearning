#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:38:07 2020

This file includes functions which are helpful to visualize the partitions and
the Q functions for the Oil and Ambulance problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from plot_rl_experiment import plot as plot_rl

def get_q_values(node):
    """
    Return all triples (state, action, q)
    
    Parameters
    ----------
    node : Node
        Initial node.

    Returns
    -------
    Recursively transverse the tree, and return all triples (state, action, q)

    """
    if node.children == None:
        return [[node.state_val, node.action_val, node.qVal]]
    else:
        q_values = []
        for c in node.children:
            q_values.extend(get_q_values(c))
        return q_values

def xy_plot_node(node):
    """
    Returns the information required to draw the partition associated with Node
    node.

    Parameters
    ----------
    node : Node
        Initial node.

    Returns
    -------
    The a collection of rectangle coordinates encoding the state-action space
    partition for the input node.

    """
    rects = []
    if node.children == None:
        rect = [node.state_val - node.radius/2, node.action_val - node.radius/2,
                0, node.radius, node.radius, node.qVal]
        rects.append(rect)
    else:
        for child in node.children:
            rects.extend(xy_plot_node(child))
    return np.array(rects)

def scatter_q_values(tree, fig=None, animated=False):
    """
    Plot the Q function as a scatter plot.

    Parameters
    ----------
    tree : Tree
        A Tree instance.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    animated : bool, optional
        Set this flag when making a video. The default is False.

    Returns
    -------
    Scatter plot of the Q function.

    """
    if not fig:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=30., azim=-120)
    else:
        ax = fig.gca()
    q_values = np.array(get_q_values(tree.head))
    return ax.scatter(q_values[:,0], q_values[:,1], q_values[:,2],
                      animated=animated)
    
def bar_q_values(tree, fig=None, animated=False):
    """
    Plot the Q function as a bar graph.

    Parameters
    ----------
    tree : Tree
        A Tree instance.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    animated : bool, optional
        Set this flag when making a video. The default is False.

    Returns
    -------
    Bar graph of the Q function.

    """
    if not fig:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=30., azim=-120)
    else:
        ax = fig.gca()
    # Draw the partition
    bars = xy_plot_node(tree.head)
    return ax.bar3d(bars[:,0],bars[:,1],bars[:,2],bars[:,3],bars[:,4],bars[:,5],
                    alpha=0.5,animated=animated,color='r')

def plot_partition_bar_q(tree, fig=None, file_name=None):
    """
    Plot the 2D partition and the Q function bar graph side by side.

    Parameters
    ----------
    tree : Tree
        A Tree instance.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    file_name : string, optional
        Pass this argument to store the resulting image. The default is None
        (no image is stored).

    Returns
    -------
    None.

    """
    if not fig:
        fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    plt.figure(fig.number)
    fig.sca(ax1)
    # Plot the partition
    tree.plot(0)
    # Plot the bar graph
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=30., azim=-120)
    bars = xy_plot_node(tree.head)
    ax2.bar3d(bars[:,0],bars[:,1],bars[:,2],bars[:,3],bars[:,4],bars[:,5],
                alpha=0.5, color='r')
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300)
        
def plot_rollout(agent, envClass, envParams, epLen=None, fig=None, ax=None):
    """
    Runs an episode of envClass(**envParams) choosing actions using Agent agent.
    Plots the (state, action) pairs on the state-action space, and returns the
    cumulative reward.
    
    This is a helper function for the inspect_agent.py tool.

    Parameters
    ----------
    agent : Agent class instance
        An AQL or SPAQL agent.
    envClass : Environment class
        An oil or ambulance problem class.
    envParams : dict
        The environment initialization parameters.
    epLen : int, optional
        Episode length. The default is None.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    ax : plt.Axes, optional
        A matplotlib axes instance. The default is None.

    Returns
    -------
    epReward : float
        Cumulative reward of the episode.

    """
    if len(agent.tree_list) > 1:
        return
    if not epLen:
        epLen = agent.epLen
    if not fig:
        fig = plt.figure(figsize=(6,6))
    if not ax:
        ax = fig.gca()
    agent.tree.plot(0)
    env = envClass(**envParams)
    env.reset()
    state = env.state
    epReward = 0
    for i in range(epLen):
        label = i+1
        action = agent.pick_action(state, i)
        ax.annotate(str(label), (state, action))
        reward, state, pContinue = env.advance(action)
        epReward += reward
    return epReward
        
def plot_multi_partition_bar_q(tree_list, fig=None, file_name=None):
    """
    Plot the partition for the AQL agents (one partition per time step).

    Parameters
    ----------
    tree_list : list
        List of Tree instances.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    file_name : string, optional
        Pass this argument to store the resulting image. The default is None
        (no image is stored).

    Returns
    -------
    None.

    """
    if not fig:
        fig = plt.figure(figsize=(12,6))
    plt.figure(fig.number)
    n = len(tree_list)
    # ax1 = fig.add_subplot(2, n, 1)
    for i in range(n):
        ax1 = fig.add_subplot(2, n, i+1)
        fig.sca(ax1)
        # Plot the partition
        tree = tree_list[i]
        tree.plot(0)
        # Plot the bar graph
        ax2 = fig.add_subplot(2, n, n+i+1, projection="3d")
        ax2.view_init(elev=30., azim=-120)
        bars = xy_plot_node(tree.head)
        ax2.bar3d(bars[:,0],bars[:,1],bars[:,2],bars[:,3],bars[:,4],bars[:,5],
                    alpha=0.5, color='r')
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300)
        
def plot_learning_curve_bar(rewards, tree, fig=None, file_name=None):
    """
    Plots the learning curve and Q function bar graph side by side.

    Parameters
    ----------
    rewards : list
        Evolution of rewards along training.
    tree : Tree
        Tree instance.
    fig : plt.Figure, optional
        A matplotlib figure. The default is None.
    file_name : string, optional
        Pass this argument to store the resulting image. The default is None
        (no image is stored).

    Returns
    -------
    None.

    """
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(121)
    # Plot the learning curve
    ax.plot(range(1, len(rewards)+1), rewards, linewidth=1)
    
    # Get the current figure
    fig = plt.gcf()
    ax1 = fig.add_subplot(122, projection="3d")
    ax1.view_init(elev=30., azim=-120)
    
    # Plot the partition
    # tree.plot(0)
    # Plot the bar graph
    bar_q_values(tree, fig=fig)
    if file_name:
        plt.savefig(file_name, dpi=300)
    
if __name__ == "__main__":
    from tree import Tree
    import matplotlib.animation as animation
    
    # Plot the tree
    bar_q_values(Tree(1))
    
    # Plot the tree using an existing figure
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=30., azim=-120)
    bar_q_values(Tree(2), fig)
    
    # Make a video
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=30., azim=-120)
    
    ims = []
    for i in range(60):
        im = bar_q_values(Tree(i), fig)
        ims.append([im])
    #     plt.savefig("photos/{}.png".format(i))
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    # plt.scf(fig)
    ani.save("q_value_animation.mp4")