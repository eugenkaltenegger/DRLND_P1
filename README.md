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

To create a conda workspace with the dependencies required for this repository (listed in `requirements.txt`) run the following command in the base directory of this repository:
```bash
conda create --name kalteneger_p1_navigation --file requirements.txt
```
This conda environment has to be activated with the following command:
```bash
conda activate kalteneger_p1_navigation
```

With the active conda environment the preparation to run the code is complete.

###Execution

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

###Findings

### Improvements
In the task description a benchmark solving the environment in about 1800 episodes is provided.
The solution presented in this repository is capable to solve the environment in about ???? episodes.

The following changes could improve the learning process and resulting neuronal network:
- extended hyperparameter tuning
  an exhaustive grid search for the perfect hyperparameters could improve the learning process and the resulting neuronal network
- parallel training for improved hyperparameter tuning
  enabling parallel training could improve the process of hyperparameter training and therefore could improve the learning process and the resulting neuronal network
- weighting the experiences
  the solution presented here selects a random batch from the replay buffer when the agent is learning from the replay buffer.
  the events in the replay buffer could be 

## Summary
