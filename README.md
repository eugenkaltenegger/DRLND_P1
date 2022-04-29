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

###Dependencies

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

This repository requires executables to run the environment and a conda workspace.
For the Banana World the executable can be downloaded [here](Banana) and for the Pixel Banana World the executable can be downloaded [here](PixelBanana).

[Banana]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
[PixelBanana]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip

The downloaded `.zip` files have to be extracted and placed in the folder `env`.
The resulting file structure should look like shown below:
```bash
├── env
│   ├── Banana_Linux
│   │   ├── Banana_Data
│   │   ├── Banana.x86
│   │   └── Banana.x86_64
│   └── PixelBanana_Linux
│       ├── Banana_Data
│       ├── Banana.x86
│       └── Banana.x86_64
```

To create a conda envirnment and install the packages required for this repository run the following command:
```bash
conda env create --file requirements.yaml
```

This conda environment has to be activated with the following command:
```bash
conda activate kalteneger_p1_navigation
```

With the active conda environment and the installed dependencies the preparation to run the code is completed.

###Execution

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

To execute the code run the following command in a terminal with the active conda environment:
```bash
python3 navigation.py <mode> <world>
```

To code provided in this repository has three operation modes:
- `tune`: hyperparameter tuning, the list of hyperparameters set in form of lists in the file `navigation.py` in the ordered dict with the name `hpr` is applied.
  The results of each hyperparameter combination are shown and finally the combination solving the environment after the least steps with the highest score is listed.
- `train`: training the agent, the agent is trained with the hyperparameters set in the file `navigation.py` in the ordered dict with the name `ph`.
  The graphs for the score and the average score over the last 100 episodes are displayed and the trained network is stored.
- `show`: showing the operation of the trained agent.
  The simulation is started with visualization and the trained agent is operating in the environment.
  This mode is for visualization purposes only.

The solution can either operate on the `Banana` environment or the `PixelBanana` environment.
The world argument has to be either:
- `Banana` (default)
- `PixelBanana`

To start the program the command could look like:
```bash
python3 navigation.py show Banana
```

###Architecture

The agent has a neuronal network which has an input of the size of the state space of the world and an output of the size of the action space.
In between the input layer and output layer a neuronal network with a dynamic number of layers each with a dynamic size is created.
The number of layers and the size of each layer are read when the agent is created.
These values can be set in the `naviagtion.py` file.
Each layer, expect the last is followed by ReLU function.

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
    self.hp["batch_size"] = 32
    self.hp["batch_frequency"] = 4
    self.hp["episodes"] = 2000
    self.hp["steps_per_episode"] = 1500
    self.hp["epsilon_start"] = 0.10
    self.hp["epsilon_end"] = 0.01
    self.hp["epsilon_factor"] = 0.95
```

With these values it was possible to solve the environment after 500 to 700 episodes.

The agent is also able to solve the `PixelBanana` environment, but for this environment the parameters have not been tweaked.

### Improvements

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
