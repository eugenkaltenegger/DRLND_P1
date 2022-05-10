# Udacity - Deep Reinforcement Learning - Project 1

## Project Description

The task of this project is to train an agent to navigate in an environment with bananas where yellow bananas shall be collected and blue bananas shall be avoided.
In this environment a reward of +1 is granted for collecting a yellow banana and a reward of -1 is granted for collecting a blue banana.
Therefore, the agent shall seek yellow bananas and shall avoid blue bananas.

The environment provides a state vector either in form of
- a vector with dimension 37 (Banana World)
- a vector with dimension 7056 (PixelBanana World)

The environment is considered as solved if the agent achieves an average score of 13 or higher in 100 consecutive runs.

The agent shall utilize the DQN architecture presented in the [DQN paper](DQN).

[DQN]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

### Dependencies

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

In order to operate this repository it is necessary to download the executables for the environment and to create a conda environment.
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

### Execution

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
