#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:20:55 2020

@author: joaopedroaraujo
"""
import gym
import numpy as np
import itertools

from src.environment import ContinuousAIGym
from tree import Tree, Node

from agents import SinglePartitionSoftmaxControlAgent, MultiPartitionAgent

class PendulumEnv(ContinuousAIGym):
    def __init__(self, scale_reward=False):
        env = gym.make('Pendulum-v0')
        epLen = 200
        super(PendulumEnv, self).__init__(env, epLen)
        
        # In case we want to scale the rewards to interval [0, 1]
        self.scale_reward = scale_reward
        
    def advance(self, action):
        """
        Since we will be receiving an action from [-1, 1] and returning an
        observation from [-1, 1]^3, we need to wrap that.
        
        The Pendulum environment has the following specs
        
        cos(theta) (state): [-1, 1]
        sin(theta) (state): [-1, 1]
        theta_dot  (state): [-8, 8]
        
        torque (action): [-2, 2]
        """
        # Wrap the action
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, int) or isinstance(action, float):
            action = [action]
        elif not isinstance(action, np.ndarray):
            print("Invalid action. Must be list or number. Returning.")
            return
        
        assert len(action) == 1
        
        # Failing this assertion caused problems in the past
        assert np.abs(action) <= 1
        reward, newState, pContinue = super(PendulumEnv, self).advance(action*2)
        assert np.abs(action) <= 1
        
        # Wrap the state. We only need to wrap the angular velocity.
        newState[2] /= 8
        assert np.abs(newState[2]) <= 1
        
        # Return the data
        return self.rescale_reward(reward), newState, pContinue
    
    def rescale_reward(self, reward):
        if not self.scale_reward: return reward
        """
        The pendulum problem is parametrized by a cost which ranges from [-16, 0].
        This function remaps it to [0, 1].
        """
        max_cost = np.pi**2 + .1*8**2 + .001*(2**2)
        new_reward = (reward + max_cost) / max_cost
        assert new_reward > 0 and new_reward < 1
        return new_reward
        

class PendulumNode(Node):
    def __init__(self, qVal, num_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        super(PendulumNode, self).__init__(qVal, num_visits, num_splits, state_val, action_val, radius)
        
    def split_node(self):
        """
        Works for any dimensional spaces.
        """
        self.children = []
        
        # Split the state space
        state_centers = []
        nbits = len(self.state_val)
        words = list(itertools.product([0, 1], repeat=nbits))
        for i in range(len(words)):
            word = words[i]
            center = []
            for j in range(len(word)):
                center.append(self.state_val[j] + (-1)**int(word[j]) * self.radius*(1/2))
            state_centers.append(center)
            # self.children.append(PendulumNode(self.qVal, self.num_visits,
            #                                   self.num_splits+1, center,
            #                                   self.action_val, self.radius*(1/2)))
            
        # Split the action space
        action_centers = []
        nbits = len(self.action_val)
        words = list(itertools.product([0, 1], repeat=nbits))
        for i in range(len(words)):
            word = words[i]
            center = []
            for j in range(len(word)):
                center.append(self.action_val[j] + (-1)**int(word[j]) * self.radius*(1/2))
            action_centers.append(center)
            
        for state in state_centers:
            for action in action_centers:
                self.children.append(PendulumNode(self.qVal, self.num_visits,
                                                  self.num_splits+1, np.array(state),
                                                  np.array(action), self.radius*(1/2)))
        
        return self.children
    
class PendulumTree(Tree):
    def __init__(self, epLen=200, nodeClass=PendulumNode, nodeClassArgs={}):
        self.epLen = epLen
        self.head = nodeClass(**nodeClassArgs)
    
    def plot_node(self, node, ax):
        raise(NotImplementedError)
        
    def state_within_node(self, state, node):
        """
        State is a 3D vector, and so is node.state_val
        
        If this is as simple as adding an "all" call, then there is no need for
        a separate function...
        """
        return all(np.abs(state - node.state_val) <= node.radius)
    
class MultiPartitionPendulum(MultiPartitionAgent):
    def __init__(self, epLen, numIters, scaling, video=False):
        super(MultiPartitionPendulum, self).__init__( epLen, numIters, scaling, video)
        
        self.tree_list = []
        for h in range(epLen):
            tree = PendulumTree(epLen=epLen, nodeClass=PendulumNode,
                                 nodeClassArgs={'qVal': epLen,
                                               'num_visits': 0,
                                               'num_splits': 0,
                                               'state_val': np.array([0, 0, 0]),
                                               'action_val': np.array([0]),
                                               'radius': 1})
            self.tree_list.append(tree)
        

class PendulumAgent(SinglePartitionSoftmaxControlAgent):
    def __init__(self, epLen=200, numIters=500, scaling=0,
                 terminal_state=np.array([1, 0, 0]), lam=np.inf, exponent=2):
        super(PendulumAgent, self).__init__(epLen, numIters, scaling, terminal_state, lam)
        self.tree = PendulumTree(epLen=epLen, nodeClass=PendulumNode,
                                 nodeClassArgs={'qVal': epLen,
                                               'num_visits': 0,
                                               'num_splits': 0,
                                               'state_val': np.array([0, 0, 0]),
                                               'action_val': np.array([0]),
                                               'radius': 1})
        self.tree_list = [self.tree]
        
        self.exponent = exponent
        
    def pair_within_node(self, state, action, node):
        d = max([max(np.abs(state - node.state_val)), max(np.abs(action - node.action_val))])
        return d <= node.radius
    
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
        # if timestep == self.epLen - 1: # terminal state
        if self.lam != np.inf:
            vFn *= np.exp(-(np.linalg.norm(obs - self.opt)/self.lam)**2)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= (2**self.exponent)**active_node.num_splits:
            active_node.split_node()
            
class PendulumAgentTerminalState(PendulumAgent):
    def __init__(self, **kwargs):
        super(PendulumAgentTerminalState, self).__init__(**kwargs)

if __name__ == "__main__":
    p = PendulumEnv()
    tree = PendulumTree(nodeClassArgs={'qVal': 200,
                                       'num_visits': 0,
                                       'num_splits': 0,
                                       'state_val': np.array([0, 0, 0]),
                                       'action_val': np.array([0]),
                                       'radius': 1})
    tree.head.split_node()
    node1 = tree.head.children[1]
    node14 = tree.head.children[14]
    states = np.zeros([200, 3])
    for i in range(200):
        # reward, states[i,:], pC = p.advance((-1)**i)
        p.env.state[0] = 3.14/16 * i 
        p.env.render()
    p.env.close()