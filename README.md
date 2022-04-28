# Udacity - Deep Reinforcement Learning - Project 1

## Project Description

The task of this project is to train an agent to navigate in an environment with bananas where yellow bananas shall be collected and blue bananas shall be avoided.
In this environment a reward of +1 is granted for collecting a yellow banana and a reward of -1 is granted for collecting a blue banana.
Therefore, the agent shall seek yellow bananas and shall avoid blue bananas.
In the environment a state space of the dimension 37 is provided while an action space of the dimension 4 is provided.
The environment is considered as solved if the agent achieves an average score of 13 or higher in 100 consecutive runs.

The agent shall utilize the DQN architecture presented in the [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

## Solution Description



### Dependencies

### Execution

### Findings


The solution in this repository contains the following hyperparameters to tune:
- hidden layers structure
- replay buffer size and replay buffer batch leaning size and frequency
- number of steps per episode
- random action selection

### Improvements
In the task description a benchmark solving the environment in about 1800 episodes is provided.
The solution presented in this repository is capable to solve the environment in about ???? episodes.

The following changes could improve the learning process and resulting neuronal network:
- extended hyperparameter tuning
  an exhausive grid search for the perfect hyperparamterts could improve the learning process and the resulting neuronal network
- parallel training for improved hyperparameter tuning
  enableling parallel training could improve the process of hyperparameter training and therefore could improve the learning process and the resulting neuronal network
- weighting the experiences
  the solution presented here selects a random batch from the replay buffer when the agent is learning from the replay buffer.
  the events in the replay buffer could be 

## Summary
