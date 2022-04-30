#Udacity - Deep Reinforcement Learning - Project 1

##Project Description

The task of this project is to train an agent to navigate in an environment with bananas where yellow bananas shall be collected and blue bananas shall be avoided.
In this environment a reward of +1 is granted for collecting a yellow banana and a reward of -1 is granted for collecting a blue banana.
Therefore, the agent shall seek yellow bananas and shall avoid blue bananas.

The environment provides a state vector either in form of
- a vector with dimension 37 (Banana World)
- a vector with dimension 7056 (PixelBanana World)

The environment is considered as solved if the agent achieves an average score of 13 or higher in 100 consecutive runs.

The agent shall utilize the DQN architecture presented in the [DQN paper](DQN).

[DQN]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

##Solution Description

The environment is solved with the code and the agent in this repository.
The solution is build in various python files.
How to install the program and the dependencies is described in [Dependencies](###Dependencies) and how operate the program is described in [Execution](###Execution).

In order to solve the environment the agent trains a neuronal network with batch learning.
For this reason the agent operates in the environment and collects experiences and learns these experiences in batches.
To provide various experiences the agent sometimes selects a random action.
The probability to select a random action decreases during the learning process.

###General Architecture

Generally DQN consist of the main parts.
Firstly a neuronal network, secondly a buffer which stores experiences of the agent in the environment.

####Neuronal Network
The neuronal network is a group of layers of neurons and connections between the neurons of one layer to the neurons of the next layer.
The input layer is the entry point of information and the output layer is the exit point of the network where a decision is made.
Information that is passed into the network activates the neurons in the input layer which again activates the next layer and so on until the output layer is activated and a decision was made.
The intensity of the activation depends on the weights stored in the connection of the neurons.
During the training of the agent these weights are adopted to achieve the desired activation in the output layer corresponding to the input at the input layer.

####Buffer
The buffer stores experiences which the agent makes in the environment.
This includes the state of the environment, the action of the agent, the reward the agent gains and the resulting state.
The agent calls requests a batch of such tuples after a few steps and learn all of them at once.
This strategy avoids that single actions are repeated to often and therefore are trained more than other actions.

###Specific Architecture

The solution for the environment introduced above is split up into various files:

- `navigation.py`:
  The entry point for the code execution.
  The main logic of the execution modes (`tune`,`train` and `show` - for more details please have a look at the README.md file) is in this file.
  THis is the location of the hyperparameters and the list of hyperparameters to tune.
- `src/agent.py`:
  The agent is the part which contains the model and the buffer which build the heart of DQN.
  The agent can save and load the current state of den neuronal network.
- `src/buffer.py`:
  The buffer is one of the two main features of DQN.
  The agent stores experiences in the buffer and after a few step the agent learn a batch of experiences from the buffer.
  The batch which the agent learn is drawn randomly from the buffer.
- `src/environment.py`:
  In this file the environment API is reduced to those features which are necessary for this solution.
  The functions `state_size()` and `action_size()` allow swapping between the Banana and the PixelBanana world without any other changes in the code (still changes in the hyperparameter might be required to efficiently solve the environment)
- `src/model.py`:
  This file is one of the two main features of DQN.
  Here the neuronal network which the agent utilized is created.
  The structure of the neuronal network (the number of layers and the size of each layer) is passed to the agent when creating the agent.
  After each layer of the neuronal network a ReLU layer is added, expect for the last layer (output layer) of the neuronal network.

The agent has a neuronal network which has an input of the size of the state space of the world and an output of the size of the action space.
In between the input layer and output layer a neuronal network with a dynamic number of layers each with a dynamic size is created.
The number of layers and the size of each layer are read when the agent is created.
These values can be set in the `naviagtion.py` file.
Each layer, expect the last is followed by ReLU function.

The agent shown in this repository learns not after each step, it learns after a given number of steps the results from a few steps.
This is called batch learning and a key feature of the DQN architecture.

###Findings

While solving this exercise various configurations have been tried.

The agent was able to learn fast with three layers where the layer size decreases from layer to layer.
Layersizes between 512 and 256 work very well.
The buffer size had a small impact on the agent's performance while the batch size and the number of steps after which the agent learns the batch had a higher impact.
A high number of steps per episode help to ensure that the agent is provided with sufficient experience per episode.
If the number of steps per episode is too low the training is not working as intended.
A high starting epsilon value ensures that the agent starts with random actions.
The probability for random actions shrinks after each episode until epsilon reaches its minimum.

The following parameters can be part of the hyperparameter tuning process:
- `layers`: number and size of layers, as a list
- `buffer_size`: size of the replay buffer
- `batch_size`: size of a batch which the agent will lean
- `batch_frequency`: number of steps after which the agent learns a batch
- `episodes`: number of episodes given to solve the enironment
- `steps_per_episode`: steps per episode
- `epsilon_start`: epsilon (defining the probability for random actions) start value 
- `epsilon_end`: epsilon (defining the probability for random actions) end value
- `epsilon_factor`: epsilon (defining the probability for random actions) reduction factor applied after each episode

The following hyperparameter work very well for the `Banana` environment (not `PixelBanana` environment):
```python
    self.hp["layers"] = [256, 128, 64]
    self.hp["buffer_size"] = 2048
    self.hp["batch_size"] = 16
    self.hp["batch_frequency"] = 4
    self.hp["episodes"] = 2000
    self.hp["steps_per_episode"] = 2000
    self.hp["epsilon_start"] = 0.10
    self.hp["epsilon_end"] = 0.01
    self.hp["epsilon_factor"] = 0.99
```

With these values it was possible to solve the environment after 500 to 700 episodes.

The agent is also able to solve the `PixelBanana` environment, but for this environment the parameters have not been tweaked.

The following pictures have been created with the parameters shown above:
- `scores_absolute`: shows the scores of each episode
- `scores_relative`: shows the average score over 100 episodes, calculated after each episode (less noisy than `scores absolute`)

###Improvements

In the task description a benchmark solving the environment in about 1800 episodes is provided.
The solution presented in this repository is capable to solve the environment in about 500 to 700 episodes with the parameters provided above.

The following changes could improve the learning process and resulting neuronal network:
- Extended Hyperparameter Tuning:
  An exhaustive grid search for the best hyperparameters could improve the learning process and the resulting neuronal network.
- Parallel Training:
  Enabling parallel training could improve the process of hyperparameter tuning and therefore could improve the learning process and the resulting neuronal network.
- Changing the Model Architecture:
  The solution as presented here is utilizing the ReLU function for the model.
  Trying other functions and experimenting with the model architecture could improve the quality of the agent and the resulting neuronal network.
- Weighting the Experiences:
  The solution presented here selects a random batch from the replay buffer when the agent is learning from the replay buffer.
  Weighting the experiences in the replay buffer could improve the learning process and the resulting neuronal network.

## Summary

The exercise was interesting and fun to solve.
The deep reinforcement learning part was not difficult to solve given the code available in the various udacity workspaces.
The most difficult part was to get pytorch working on a rather new graphics card and creating a reproducible conda environment.