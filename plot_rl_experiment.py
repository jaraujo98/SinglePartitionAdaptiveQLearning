#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:03:03 2020

Based on chart_utils.py from SimpleRL

https://github.com/david-abel/simple_rl
"""

#### Graphics stuff

import matplotlib
import matplotlib.pyplot as pyplot
colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

if __name__ == "__main__":
    pyplot.style.use("fivethirtyeight")
    # Set font.
    font = {'size':14}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42

EVERY_OTHER_X = False
CUSTOM_TITLE = None
X_AXIS_LABEL = None
Y_AXIS_LABEL = None
X_AXIS_START_VAL = 0
X_AXIS_INCREMENT = 1
Y_AXIS_END_VAL = None
COLOR_SHIFT = 0

#### All else

import numpy as np
import math

def plot_rl_exp(*result_mat, names=None, ci=True, **kwargs):
    """
    Given matrices result_mat, where each column corresponds to a different trial
    (random seed) of an experiment, plots the average and the confidence interval
    (95%). If ci is False, then it plots the standard error.

    Parameters
    ----------
    *result_mat : A tuple of 2D matrices
        A matrix where each column corresponds to a time series of average
        returns. If more than one matrix is passed, it plots them all on the same
        plot
    names : list
        Optional: list of names of the several matrices
    ci : Boolean
        If true, plot the 95% confidence interval. Else, plot the standard error

    Returns
    -------
    None.

    """
    if names != None:
        # If names are specified, we require one for each matrix
        assert len(names) == len(result_mat)
    else:
        names = ['']
    
    mul = 1
    if ci:
        mul = 1.96

    avgs = []
    cis = []
    for mat in result_mat:
        average = mat.mean(axis=1)
        avgs.append(average)
        std = mat.std(axis=1)
        ci = mul * std / math.sqrt(mat.shape[1])
        cis.append(ci)
        
    plot(avgs, None, names, conf_intervals=cis, cumulative=True, **kwargs)
        
def plot(results, experiment_dir, agents, plot_file_name="", conf_intervals=[], use_cost=False, cumulative=False, episodic=True, open_plot=True, track_disc_reward=False, add_legend=True, fig=None):

    '''
    Based on
    https://github.com/david-abel/simple_rl/blob/master/simple_rl/utils/chart_utils.py
    
    Args:
        results (list of lists): each element is itself the reward from an episode for an algorithm.
        experiment_dir (str): path to results.
        agents (list): each element is an agent that was run in the experiment.
        plot_file_name (str)
        conf_intervals (list of floats) [optional]: confidence intervals to display with the chart.
        use_cost (bool) [optional]: If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        cumulative (bool) [optional]: If true, plots are cumulative cost/reward.
        episodic (bool): If true, labels the x-axis "Episode Number". Otherwise, "Step Number". 
        open_plot (bool)
        track_disc_reward (bool): If true, plots discounted reward.
        add_legend (bool)

    Summary:
        Makes (and opens) a single reward chart plotting all of the data in @data.
    '''

    # Set x-axis labels to be integers.
    from matplotlib.ticker import MaxNLocator
    if fig:
        ax = fig.gca()
    else:
        ax = pyplot.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Some nice markers and colors for plotting.
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v','|']

    x_axis_unit = "episode" if episodic else "step"

    # Map them to floats in [0:1].
    # colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
    # colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]

    # Puts the legend into the best location in the plot and use a tight layout.
    pyplot.rcParams['legend.loc'] = 'best'

    # Negate everything if we're plotting cost.
    if use_cost:
        results = [[-x for x in alg] for alg in results]

    # Make the plot.
    print_prefix = "\nAvg. cumulative reward" if cumulative else "Avg. reward"
    # For each agent.
    for i, agent_name in enumerate(agents):

        # Add figure for this algorithm.
        agent_color_index = i
        agent_marker_index = agent_color_index
        
        # Grab new color/marker if we've gone over.
        if agent_color_index >= len(colors):
            agent_color_index = agent_color_index % len(colors)
        if agent_marker_index >= len(markers):
            agent_marker_index = agent_marker_index % len(markers)
        
        series_color = colors[agent_color_index]
        series_marker = markers[agent_marker_index]
        y_axis = results[i]
        x_axis = list(drange(X_AXIS_START_VAL, X_AXIS_START_VAL + len(y_axis) * X_AXIS_INCREMENT, X_AXIS_INCREMENT))

        # Plot Confidence Intervals.
        if conf_intervals != []:
            alg_conf_interv = conf_intervals[i]
            top = np.add(y_axis, alg_conf_interv)
            bot = np.subtract(y_axis, alg_conf_interv)
            pyplot.fill_between(x_axis, top, bot, facecolor=series_color, edgecolor=series_color, alpha=0.25)
            print("\t" + str(agents[i]) + ":", round(y_axis[-1], 5) , "(conf_interv:", round(alg_conf_interv[-1], 2), ")")

        marker_every = max(int(len(y_axis) / 100), 1)
        pyplot.plot(x_axis, y_axis, color=series_color, marker=series_marker,
                    label=agent_name, markevery=marker_every,
                    linewidth=1, markersize=0)
        if add_legend:
            pyplot.legend()
    
    # Configure plot naming information.
    unit = "Cost" if use_cost else "Reward"
    plot_label = "Cumulative" if cumulative else "Average"

    disc_ext = "Discounted " if track_disc_reward else ""

    # Axis labels.
    x_axis_label = X_AXIS_LABEL if X_AXIS_LABEL is not None else x_axis_unit[0].upper() + x_axis_unit[1:] + " Number"
    y_axis_label = Y_AXIS_LABEL if Y_AXIS_LABEL is not None else plot_label + " " + unit
    
    if not Y_AXIS_END_VAL in [0, None]:
        pyplot.ylim((0, Y_AXIS_END_VAL))
    
    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    if EVERY_OTHER_X:
        pyplot.xticks(range(X_AXIS_START_VAL, len(x_axis) * X_AXIS_INCREMENT + X_AXIS_START_VAL, X_AXIS_INCREMENT * 2))

    pyplot.ylabel(y_axis_label)
    pyplot.grid(True)
    pyplot.tight_layout() # Keeps the spacing nice.
    
    if plot_file_name != "":
        pyplot.savefig(plot_file_name, dpi=300)
    
    if open_plot:
        pyplot.show()

    # Clear and close.
    # pyplot.cla()
    # pyplot.close()
    
import decimal
def drange(x_min, x_max, x_increment):
    '''
    Args:
        x_min (float)
        x_max (float)
        x_increment (float)

    Returns:
        (generator): Makes a list.

    Notes:
        A range function for generating lists of floats.
        Based on code from stack overflow user Sam Bruns:
            https://stackoverflow.com/questions/16105485/unsupported-operand-types-for-float-and-decimal
    '''
    x_min = decimal.Decimal(x_min)
    while x_min < x_max:
        yield float(x_min)
        x_min += decimal.Decimal(str(x_increment))
        
if __name__ == "__main__":
    pass