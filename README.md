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

While solving this exercise various configurations have been tried.

The agent was able to learn fast with three layers where the layer size decreases from layer to layer.
Layersizes between 512 and 256 work very well.
The buffer size had a small impact on the agent's performance while the batch size and the number of steps after which the agent learns the batch had a higher impact.
A high number of steps per episode help to ensure that the agent is provided with sufficient experience per episode.
If the number of steps per episode is too low the training is not working as intended.
A high starting epsilon value ensures that the agent starts with random actions.
The probability for random actions shrinks after each episode until epsilon reaches its minimum.

The following hyperparameter work very well:
```python
    self.hp["layers"] = [256, 128, 64]
    self.hp["buffer_size"] = 2048
    self.hp["batch_size"] = 32
    self.hp["batch_frequency"] = 16
    self.hp["episodes"] = 2000
    self.hp["steps_per_episode"] = 1500
    self.hp["epsilon_start"] = 0.10
    self.hp["epsilon_end"] = 0.01
    self.hp["epsilon_factor"] = 0.99
```

With these values it was possible to solve the environment after 600 to 700 episodes.

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
