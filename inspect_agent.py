#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:35:14 2020

This is a curses tool to inspect the agentData.pk files. There are two ways to
start it

    1. Call it with an argument
    
        python inspect_agent.py agentData.pk
        
    2. Call it without arguments
    
        python inspect_agent.py
        
The second option will open a file selection dialog, which can be used to select
and agentData.pk file. Calling it with an argument will open the file directly.

Each option on the start menu does the following:
    
    1. Select agent:
        Each .pk file may contain several agents. Use this option to select one.
        
    2. Plot:
        *After an agent has been selected*, this option opens another menu,
        described below.
        
    3. Load new file:
        Opens a file selection dialog to select another .pk file. This resets
        agent choice (see option 1)
        
    4. Quit:
        Exit back to terminal.
        
The plot menu has the following options:
    
    WARNING: This tool is experimental, and therefore still has bugs. One of them
    is that it is not possible to close figures after they have been opened. The
    only way to close them is to quit the program (option 4. in the main menu).
    
    1. Plot learning curve and partition (Single Partition agents only):
        Opens a new figure with the learning curve and partition of the currently
        selected agent.
        
    2. Plot Single Partition and Q-Function:
        Plots the partition and Q function bar graph for a SPAQL agent.
        
    3. Plot Multi Partition and Q-Function:
        Same as 2., but for AQL agents.
        
    4. Rollout:
        SPAQL agents only. Plots the path in state-action space of an agent
        rollout.

        WARNING: Since agents do not record any information regarding the
        environment on which they were trained, before using the Rollout option
        it is necessary to manually edit the environment definition of the
        plot_callback function. Examples are provided in the source code below
        (lines 164 to 177).
"""

import sys
import os
from subprocess import Popen, PIPE
import pickle
import curses
import numpy as np

# For the file dialog
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.wm_withdraw()

# Only now import pyplot modules
import draw_q_function3d
import matplotlib.pyplot as plt

# Import the environments
import envs

# Some useful commands
clear = lambda oScreen: oScreen.clear()
cprintbf = lambda oScreen, x: oScreen.addstr(x,curses.A_BOLD)
cprint = lambda oScreen, x: oScreen.addstr(x)

# Screen sizes
def stty_size():
    p = Popen(['stty', 'size'], stdout=PIPE, stderr=PIPE)
    p.wait()
    output, _ = p.communicate()
    return output

screen_size = stty_size().split(b" ")
heigth = int(screen_size[0].strip())
width = int(screen_size[1].strip())

# Helper functions to deal with the agents
def select_agent(oScreen, agents, rewards):
    index = -1234
    dc = len("Which one do you want to choose (enter a number and press enter)? ") # delta column
    while index < 0 or index > len(agents):
        clear(oScreen)
        cprint(oScreen, f'''

There are {len(agents)} agents available.

The best one (reward = {np.max(rewards[-1,:])}) is {np.argmax(rewards[-1,:])}.
The worst one (reward = {np.min(rewards[-1,:])}) is {np.argmin(rewards[-1,:])}.

Which one do you want to choose (enter a number and press enter)?
''')
        if index != -1234:
            oScreen.addstr(7, dc+4, 'Invalid!')
        curses.echo()
        s = oScreen.getstr(7, dc)
        curses.noecho()
        try:
            index = int(s)
        except:
            pass
    return index

def plot_callback(oScreen, agent, rewards, agent_index):
    c = 1
    dc = len("        4. Rollout ")
    reward = None
    while c:
        clear(oScreen)
        cprint(oScreen, f'''
          Agent {agent_index} is selected.\n''')
          
        cprint(oScreen, '''
        1. Plot learning curve and partition (Single Partition agents only)
        2. Plot Single Partition and Q-Function
        3. Plot Multi Partition and Q-Function
        4. Rollout
        0. Go back\n''')
        
        if reward:
            oScreen.addstr(6, dc, f"(reward={reward:.2f})")
        
        oEvent = oScreen.getch()
        if oEvent == ord("0"):
            c = 0
        elif oEvent == ord("1"):
            fig = plt.figure(figsize=(12,6))
            draw_q_function3d.plot_learning_curve_bar(rewards, agent.tree, fig=fig)
            fig.canvas.draw()
            plt.pause(0.001)
        elif oEvent == ord("2"):
            fig = plt.figure(figsize=(12,6))
            draw_q_function3d.plot_partition_bar_q(agent.tree, fig=fig)
            fig.canvas.draw()
            plt.pause(0.001)
        elif oEvent == ord("3"):
            fig = plt.figure(figsize=(12,6))
            draw_q_function3d.plot_multi_partition_bar_q(agent.tree_list, fig=fig)
            fig.canvas.draw()
            plt.pause(0.001)
        elif oEvent == ord("4"):
            fig = plt.figure(figsize=(6,6))
            # envClass = envs.OilLaplace
            # envArgs = dict(epLen=5,
            #                starting_state=0,
            #                lam=10)
            # reward = draw_q_function3d.plot_rollout(agent, envs.OilLaplace,
            #                           dict(epLen=5, starting_state=0, lam=10),
            #                           fig=fig)
            envClass = envs.Ambulance
            envArgs = dict(epLen=5,
                           starting_state=0.5,
                           arrivals=envs.uniform,
                           alpha=0)
            reward = draw_q_function3d.plot_rollout(agent, envClass,
                                      envArgs, fig=fig)
            fig.canvas.draw()
            plt.pause(0.001)

    return '0'

def load_from_file(path=None):
    # Load the data
    with open(path, 'rb') as f:
        data = pickle.load(f)
        agents = data['agents']
        rewards = data['rewards']
        arms = data['arms']
    return agents, rewards, arms

def open_file_dialog():
    path = filedialog.askopenfilename()
    root.update_idletasks()
    root.update()
    return path
            
# Main loop
def main(oScreen):
    curses.noecho()
    curses.curs_set(0)
    oScreen.keypad(1)
    
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = open_file_dialog()
        if file_path == '': quit()
        
    agents, rewards, arms = load_from_file(file_path)

    agent_index = -1
    agent = None
    c = 1
    while c:
        clear(oScreen)
        cprintbf(oScreen, ' --- Agent inspector ---'.center(width))
        cprint(oScreen, f'''
          Reading from file {file_path}
          {len(agents)} agents loaded.\n''')
          
        if agent:
            cprint(oScreen, f'''
          Agent {agent_index} is selected.\n''')
        else:
            cprint(oScreen, f'''
          No agent selected.\n''')
          
        cprint(oScreen, '''
        1. Select agent
        2. Plot
        3. Load new file
        0. Quit\n''')
        
        oEvent = oScreen.getch()
        if oEvent == ord("0"):
            c = 0
        elif oEvent == ord("1"):
            agent_index = select_agent(oScreen, agents, rewards)
            agent = agents[agent_index]
        elif oEvent == ord("2") and agent:
            plot_callback(oScreen, agent, rewards[:,agent_index], agent_index)
        elif oEvent == ord("3"):
            new_file_path = open_file_dialog()
            if new_file_path != '':
                file_path = new_file_path
                agents, rewards, arms = load_from_file(file_path)
                agent_index = -1
                agent = None
        
# Initialize curses
# oScreen = curses.initscr()
curses.wrapper(main)
root.destroy()
# curses.endwin()