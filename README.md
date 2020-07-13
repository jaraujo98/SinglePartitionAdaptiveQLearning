# SinglePartitionAdaptiveQLearning

Code for the experiments on the papers "Single-partition adaptive Q-learning" and "Control with adaptive Q-learning".

For any questions, comments, or bugs, feel free to open an issue!

## Introduction

Single partition adaptive Q-learning (SPAQL) is an improved version of the [adaptive Q-learning (AQL)](https://github.com/seanrsinclair/AdaptiveQLearning) algorithm by Sean Sinclair for stationary problems. It arises from a line of work on efficient model-free reinforcement learning algorithms.

Credits are also due to David Abel ([simple_rl](https://github.com/david-abel/simple_rl), on which the plotting code was based) and Cédric Colas ([rl-difference-testing](https://github.com/flowersteam/rl-difference-testing), from where the `test_rl_difference.py` code was retrieved).

## Algorithm

SPAQL learns to maximize the cumulative reward on an environment by keeping an estimate of the Q values on a partition of the state-action space. The partition is refined in a data-driven way (as samples are collected, the state-action space regions which contain more samples are split into smaller ones, while less visited regions are kept coarsely partitioned). Exploration is balanced with exploitation by using a mix of upper confidence bounds (UCB) and Boltzmann exploration with an adaptive temperature schedule.

## Installation

Simply clone or download this repository.

The code depends on `joblib`, `pandas`, `numpy`, `matplotlib`, and `gym`. Comet.ml integration is available for some training data, but its usage is not recommended.

## Running experiments

For a quick start, run and inspect the `spaql_experiment.py` file. More advanced examples are in the `experiments` folder, which includes the code used to generate the images in the companion paper (Warning: those experiments take a long time to run, and the results occupy some Gb).

It is recommended to use the `start_wrapper` function, since it automatically handles caching of results. This way, there is no need to handle the training of agents and the plotting of results on different files. Agents are trained and cached the first time a file is run. On the second time, all relevant data is loaded on runtime, and can be processed in any way.

## Inspecting results

For a quick look into the results, the tool `inspect_agent.py` is provided. This is a curses tool to inspect the pickled files there results are stored. For more detailed instructions, read the docstring at the beginning of the file.

Functions are provided to plot learning curves, partitions, and Q function values.