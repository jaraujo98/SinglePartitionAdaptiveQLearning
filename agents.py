#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:17:26 2020

The classes in this file implement the AQL and SPAQL agents. A random agent
is also included.
"""

# Necessary imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D

from src import agent
from tree import Tree

from draw_q_function3d import bar_q_values

# Definition of the agents

class MultiPartitionAgent(agent.FiniteHorizonAgent):
    def __init__(self, epLen, numIters, scaling, video=False):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling

        # List of tree's, one for each step
        self.tree_list = []
            
        # Variables for the video
        self.fig = []
        self.frames = []
        
        # Makes a new partition for each step and adds it to the list of trees
        # Does the same for the videos
        for h in range(epLen):
            tree = Tree(epLen)
            self.tree_list.append(tree)
            if video:
                self.fig.append(plt.figure())
                ax = Axes3D(self.fig[h])
                ax.view_init(elev=30., azim=-120)
                self.frames.append([])

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree_list = []
        for h in range(self.epLen):
            tree = Tree(self.epLen)
            self.tree_list.append(tree)

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree based on current timestep
        tree = self.tree_list[timestep]
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        if timestep == self.epLen - 1:
            vFn = 0
        else:
            # Gets the next tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree_list[timestep + 1]
            new_active, new_q = new_tree.get_active_ball(newObs)
            vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy
        pass

    def split_ball(self, node):
        pass

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)

        return action

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
    
    def add_frame(self):
        for i in range(self.epLen):
            im = bar_q_values(self.tree_list[i], fig=self.fig[i], animated=True)
            self.frames[i].append([im])
        
    def export_video(self, name="q_value_animation.mp4"):
        folder = "/".join(name.split("/")[:-1])
        file_name = name.split("/")[-1].split(".")[0]
        log_dir = folder + '/' + file_name
        os.mkdir(log_dir)
        for i in range(self.epLen):
            ani = animation.ArtistAnimation(self.fig[i], self.frames[i], interval=50,
                                            blit=True, repeat_delay=1000)
            ani.save("{0}/{1}_{2}.mp4".format(log_dir, file_name, i))
    
    def show_video(self):
        for i in range(self.epLen):
            ani = animation.ArtistAnimation(self.fig[i], self.frames[i], interval=50,
                                            blit=True, repeat_delay=1000)

class RandomAgent(agent.FiniteHorizonAgent):
    def __init__(self, epLen, numIters, dim=1):
        '''
        args:
            epLen - number of steps per episode
            numIters - total number of iterations
            
        This agent returns a random number in the interval [0, 1]^dim
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.dim = dim

    def reset(self):
        pass

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        return 1

    def update_obs(self, obs, action, reward, newObs, timestep):
        pass
    
    def batch_update_obs(self, *args):
        pass

    def update_policy(self, k):
        pass

    def pick_action(self, state, timestep):
        action = np.random.random(self.dim)
        return action
    
class SinglePartitionAgent(agent.FiniteHorizonAgent):
    def __init__(self, epLen, numIters, scaling, video=False):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            
        Adaptive Agent, but which only keeps one partition.
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        tree = Tree(epLen)
        self.tree = tree
        self.tree_list.append(tree)
        
        # Variables for the video
        if video:
            self.fig = plt.figure()
            ax = Axes3D(self.fig)
            ax.view_init(elev=30., azim=-120)
            self.frames = []

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree_list = []
        tree = Tree(self.epLen)
        self.tree = tree
        self.tree_list.append(tree)

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        if timestep == self.epLen - 1:
            vFn = 0
        else:
            # Gets the same tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree
            new_active, new_q = new_tree.get_active_ball(newObs)
            vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy
        pass

    def split_ball(self, node):
        pass

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)

        return action

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
    
    def add_frame(self):
        im = bar_q_values(self.tree, fig=self.fig, animated=True)
        self.frames.append([im])
        
    def export_video(self, name="q_value_animation.mp4"):
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=50,
                                        blit=True, repeat_delay=1000)
        ani.save(name)
    
    def show_video(self):
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=50,
                                        blit=True, repeat_delay=1000)
    
class SinglePartitionSoftmaxAgent(SinglePartitionAgent):
    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            
        Adaptive Agent, but which only keeps one partition.
        '''
        super(SinglePartitionSoftmaxAgent, self).__init__(epLen, numIters, scaling)
        
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node = self.get_active_ball(obs, action) # Action may have been stochastic

        if timestep == self.epLen - 1:
            vFn = 0
        else:
            # Gets the same tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree
            new_active, new_q = new_tree.get_active_ball(newObs)
            vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()
    
    def pick_action_training(self, state, timestep=None, temperature=0.01):
        """
        Chooses an action stochastically during training. If temperature is
        around 0.01, this is the same as pick_action (greedy).
        
        Parameters
        ----------
        state : float
            State of the system.
        timestep : int, optional
            Time step within the current episode. The default is None.
        temperature : float, optional
            Boltzmann distribution temperature parameter. The default is 0.01
            (argmax).

        Returns
        -------
        action : float
            The action to perform.

        """
        nodes, qValues = self.get_all_q_values(state)
        
        qValues = np.array(qValues)
        qValues = qValues/np.max(np.abs(qValues))
        
        p = np.exp(qValues/temperature)/sum(np.exp(qValues/temperature))
        
        index = np.digitize(np.random.random(), np.cumsum(p))
        node = nodes[index]
        
        return self.pick_softmax_action(node)
    
    def pick_softmax_action(self, active_node):
        action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)
        
        return action
    
    def get_node_q_values(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return [node], [node.qVal]
        else:
            # Otherwise checks each child node
            nodes = []
            qVal = []
            for child in node.children:
                # if the child node contains the current state
                if self.tree.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qVal = self.get_node_q_values(state, child)
                    nodes.extend(new_node)
                    qVal.extend(new_qVal)
                else:
                    pass
        return nodes, qVal
    
    def get_all_q_values(self, state):
        return self.get_node_q_values(state, self.tree.head)
    
    def get_active_ball_recursion(self, state, action, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node
        else:
            # Otherwise checks each child node
            for child in node.children:
                # if the child node contains the current state
                if self.pair_within_node(state, action, child):
                    # recursively check that node for the max one, and compare against all of them
                    active_node = self.get_active_ball_recursion(state, action, child)
                    break # Assuming there is no overlapping between nodes...
        return active_node


    def get_active_ball(self, state, action):
        """
        AQL's Tree method only considers the state value when finding the
        active ball. This works when the policy is the argmax, but not when
        using softmax.

        Parameters
        ----------
        state : float
            The system's state.
        action : float
            The action which was chosen.

        Returns
        -------
        The ball that contains the (state, action) pair passed as input.

        """
        return self.get_active_ball_recursion(state, action, self.tree.head)
    
    def pair_within_node(self, state, action, node):
        d = max([np.abs(state - node.state_val), np.abs(action - node.action_val)])
        return d <= node.radius
    
class SinglePartitionInfiniteAgent(SinglePartitionAgent):
    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            
        Adaptive Agent, but which only keeps one partition.
        '''
        super(SinglePartitionInfiniteAgent, self).__init__(epLen, numIters, scaling)
        
        self.tree = Tree(epLen)
        self.tree_list = [self.tree]
        
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        new_tree = self.tree
        new_active, new_q = new_tree.get_active_ball(newObs)
        vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()
            
class SinglePartitionSoftmaxInfiniteAgent(SinglePartitionSoftmaxAgent):
    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            
        Adaptive Agent, but which only keeps one partition.
        '''
        super(SinglePartitionSoftmaxInfiniteAgent, self).__init__(epLen, numIters, scaling)
        self.tree = Tree(epLen)
        self.tree_list = [self.tree]
        
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node = self.get_active_ball(obs, action) # Here use the state-action pair (might be stochastic)
        # print(f"visited node : [{active_node.state_val}, {active_node.action_val}]")

        new_tree = self.tree
        new_active, new_q = new_tree.get_active_ball(newObs) # Here use argmax (deterministic)
        vFn = min(self.epLen, new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()
        
class SinglePartitionSoftmaxControlAgent(SinglePartitionSoftmaxAgent):
    def __init__(self, epLen, numIters, scaling, terminal_state, lam=1):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
            
        Adaptive Agent, but which only keeps one partition.
        '''
        super(SinglePartitionSoftmaxControlAgent, self).__init__(epLen, numIters, scaling)
        self.tree = Tree(epLen)
        self.tree_list = [self.tree]
        
        self.opt = terminal_state
        self.lam = lam
        
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node = self.get_active_ball(obs, action) # Here use the state-action pair (might be stochastic)
        # print(f"visited node : [{active_node.state_val}, {active_node.action_val}]")

        new_tree = self.tree
        new_active, new_q = new_tree.get_active_ball(newObs) # Here use argmax (deterministic) ; we are getting the value of the next state, so better use the best one!
        vFn = min(self.epLen, new_q)
        if timestep == self.epLen - 1: # terminal state
            vFn *= np.exp(-(np.linalg.norm(obs - self.opt)/self.lam)**2)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits:
            active_node.split_node()
    
if __name__ == "__main__":
    epLen = 5
    nEps = 100
    scaling = 0.5
    
    alpha = 0.01
    
    from src import environment
    env = environment.makeLaplaceOil(epLen, 1, 0)
    
    seed = int.from_bytes(os.urandom(4), 'big')
    seed=1
    np.random.seed(seed)
    
    # rAgent = RandomAgent(epLen, nEps)
    # random_action = rAgent.pick_action(1, 1)
    
    # spsAgent = SinglePartitionSoftmaxAgent(epLen, nEps, scaling)
    scaling = 5
    spiAgent = SinglePartitionSoftmaxInfiniteAgent(epLen, nEps, scaling)
    
    rolloutAgent = spiAgent
    state = 0
    actions = []
    flatten_actions = []
    epReward = 0
    for i in range(50):
        env.reset()
        state = env.state
        episode = []
        for j in range(5):
            action = rolloutAgent.pick_action_training(state, None, temperature=alpha)
            episode.append([state, action])
            # print(f'state : {state}')
            print(f'action : {action}')
            reward, newState, pContinue = env.advance(action)
            rolloutAgent.update_obs(state, action, reward, newState, i)
            rolloutAgent.add_frame()
            state = newState
        actions.append(episode)
        flatten_actions.extend(episode)
        print(f"final state : {state}")

    #bar_q_values(rolloutAgent.tree)
    rolloutAgent.show_video()
    
    # spAgent = SinglePartitionAgent(epLen, nEps, scaling)
    # state = 0
    # for i in range(40):
    #     new_state = rAgent.pick_action(1, 1)
    #     spAgent.update_obs(state, new_state, new_state, new_state, i)
    #     spAgent.add_frame()
        
    # #spAgent.export_video(name="spAgentTest2.mp4")
    
    # mpAgent = MultiPartitionAgent(epLen, nEps, scaling)
    # state = 0
    # for i in range(40):
    #     new_state = rAgent.pick_action(1, 1)
    #     mpAgent.update_obs(state, new_state, new_state, new_state, i % epLen)
    #     mpAgent.add_frame()
        
    # #mpAgent.export_video(name="mpAgentTest2.mp4")