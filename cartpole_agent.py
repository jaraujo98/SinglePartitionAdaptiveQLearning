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
from src import agent
from tree import Tree, Node

from agents import SinglePartitionSoftmaxControlAgent, MultiPartitionAgent

from pendulum_agent import PendulumTree as CartPoleTree

def bsig(x, upper_bound):
    m = 1/upper_bound
    return 2 / (1 + np.exp(-2*m*x)) - 1

def to_metric_space(state):
    x1 = state[0]/ 4.8
    x2 = bsig(state[1], 240)
    x3 = state[2] / (24 * np.pi / 180)
    x4 = bsig(state[3], 21)
    return np.array([x1, x2, x3, x4])

class CartPoleEnv(ContinuousAIGym):
    def __init__(self):
        env = gym.make('CartPole-v0')
        epLen = 200
        super(CartPoleEnv, self).__init__(env, epLen)
        
    def advance(self, action):
        """
        Since we will be receiving an action from {0, 1} and returning an
        observation from [-1, 1]^4, we need to wrap that.
        
        The CartPole environment has the following specs
        
        cos(theta) (state): [-1, 1]
        sin(theta) (state): [-1, 1]
        theta_dot  (state): [-8, 8]
        
        torque (action): [-2, 2]
        """
        # Wrap the action
        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action) == 1
            action = action[0]
            
        if not (action==1 or action==0):
            print("Invalid action. Must be either 0 or 1 (int). Returning.")
            return
        
        reward, newState, pContinue = super(CartPoleEnv, self).advance(action)
        
        # Wrap the state.
        newState = to_metric_space(newState)
        assert np.all(np.abs(newState) <= 1)
        
        # Return the data
        return reward, newState, pContinue
    
class CartPoleNode(Node):
    def __init__(self, qVal, num_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center -> Categorical!
        radius - radius of the node '''
        super(CartPoleNode, self).__init__(qVal, num_visits, num_splits, state_val, action_val, radius)
        
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
        if len(self.action_val) > 1:
            half = len(self.action_val)//2
            action_centers = [self.action_val[:half], self.action_val[half:]]
            assert self.action_val == action_centers[0] + action_centers[1]
        else:
            action_centers = [self.action_val]
            
        for state in state_centers:
            for action in action_centers:
                self.children.append(CartPoleNode(self.qVal, self.num_visits,
                                                  self.num_splits+1, np.array(state),
                                                  action, self.radius*(1/2)))
        
        return self.children

class MultiPartitionCartPole(MultiPartitionAgent):
    def __init__(self, epLen, numIters, scaling, video=False,
                 state_val=np.array([0, 0, 0, 0]), action_val=[0, 1]):
        super(MultiPartitionCartPole, self).__init__( epLen, numIters, scaling, video)
        
        self.tree_list = []
        for h in range(epLen):
            tree = CartPoleTree(epLen=epLen, nodeClass=CartPoleNode,
                                 nodeClassArgs={'qVal': epLen,
                                               'num_visits': 0,
                                               'num_splits': 0,
                                               'state_val': state_val,
                                               'action_val': action_val,
                                               'radius': 1})
            self.tree_list.append(tree)
            
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
        action = np.random.choice(active_node.action_val)

        return action
        
class CartPoleRandomAgent(agent.FiniteHorizonAgent):
    def __init__(self, epLen, numIters, action_val=[0, 1]):
        '''
        args:
            epLen - number of steps per episode
            numIters - total number of iterations
            
        This agent returns a random number in the interval [0, 1]
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.action_val = action_val

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
        '''Update internal policy based upon records'''
        pass

    def pick_action(self, state, timestep):
        action = np.random.choice(self.action_val)
        return action

class CartPoleAgent(SinglePartitionSoftmaxControlAgent):
    def __init__(self, epLen=200, numIters=500, scaling=0.5,
                 state_val=np.array([0, 0, 0, 0]), action_val=[0, 1],
                 terminal_state=np.array([0, 0, 0, 0]), lam=np.inf, exponent=2):
        super(CartPoleAgent, self).__init__(epLen, numIters, scaling, terminal_state, lam)
        self.tree = CartPoleTree(epLen=epLen, nodeClass=CartPoleNode,
                                 nodeClassArgs={'qVal': epLen,
                                               'num_visits': 0,
                                               'num_splits': 0,
                                               'state_val': state_val,
                                               'action_val': action_val,
                                               'radius': 1})
        self.tree_list = [self.tree]
        
        self.exponent = exponent
        
    def pair_within_node(self, state, action, node):
        # We are only concerned with the state, as the action is categorical
        d = max(np.abs(state - node.state_val)) # But we still need to check this is the node played
        return d <= node.radius and action in node.action_val
    
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
            
    def pick_action_training(self, state, timestep=None, temperature=0.01):
        """
        If temperature is around 0.01, this is the same as pick_action.
        This default value was previously set to 1, and brought many problems
        because it was assumed in the "rollout" function...

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        timestep : TYPE, optional
            DESCRIPTION. The default is None.
        temperature : TYPE, optional
            DESCRIPTION. The default is 0.01 (argmax).

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nodes, qValues = self.get_all_q_values(state)
        
        qValues = np.array(qValues)
        qValues = qValues/np.max(np.abs(qValues))
        
        p = np.exp(qValues/temperature)/sum(np.exp(qValues/temperature))
        
        index = np.digitize(np.random.random(), np.cumsum(p))
        node = nodes[index]
        
        action = np.random.choice(node.action_val)
        
        return action
    
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
        action = np.random.choice(active_node.action_val)

        return action

class CartPoleAgentTerminalState(CartPoleAgent):
    def __init__(self, **kwargs):
        super(CartPoleAgentTerminalState, self).__init__(**kwargs)

if __name__ == "__main__":
    p = CartPoleEnv()
    tree = CartPoleTree(nodeClass=CartPoleNode,
                        nodeClassArgs={'qVal': 200,
                                       'num_visits': 0,
                                       'num_splits': 0,
                                       'state_val': np.array([0, 0, 0, 0]),
                                       'action_val': [0, 1],
                                       'radius': 1})
    tree.head.split_node()
    # node1 = tree.head.children[1]
    # node14 = tree.head.children[14]
    states = np.zeros([200, 4])
    for i in range(200):
        # reward, states[i,:], pC = p.advance((1 - (-1)**i)//2)
        # p.env.state[3] = 3.14/16 * i 
        p.env.render()
    p.env.close()