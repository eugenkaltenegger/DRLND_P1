import itertools
import logging

import numpy
import sys
import torch
from collections import OrderedDict
from collections import deque
from matplotlib import pyplot

from src.agent import Agent
from src.environment import Environment


class Navigation:

    def __init__(self):
        logging.StreamHandler.terminator = ""

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.world = None
        self.environment = None
        self.agent = None

        logging.info("\rDEVICE: {}\n".format(self.device))

        self.hp = OrderedDict()  # hyperparameters
        self.hp["layers"] = [256, 128, 64]
        self.hp["buffer_size"] = 2048
        self.hp["batch_size"] = 32
        self.hp["batch_frequency"] = 16
        self.hp["episodes"] = 2000
        self.hp["steps_per_episode"] = 1500
        self.hp["epsilon_start"] = 0.10
        self.hp["epsilon_end"] = 0.01
        self.hp["epsilon_factor"] = 0.99

        self.hpr = OrderedDict()  # hyperparameters_range
        self.hpr["layers"] = [[256, 128, 64]]
        self.hpr["buffer_size"] = [2048]
        self.hpr["batch_size"] = [32]
        self.hpr["batch_frequency"] = [16]
        self.hpr["episodes"] = [4000]
        self.hpr["steps_per_episode"] = [2000]
        self.hpr["epsilon_start"] = [1.00]
        self.hpr["epsilon_end"] = [0.01]
        self.hpr["epsilon_factor"] = [0.99]

    def setup(self, mode, world):
        self.world = world
        if mode == "train":
            logging.info("\rMODE: training\n")
            absolute_scores, average_scores = self.train(filename="agent_state.pth")
            self.plot(scores=absolute_scores, filename="absolute_scores.png")
            self.plot(scores=average_scores, filename="average_scores.png")

        if mode == "tune":
            logging.info("\rMODE: training\n")
            self.tune()

        if mode == "show":
            logging.info("\rMODE: show\n")

    def create_agent(self):
        logging.info("\rcreating agent\n")
        return self.reset_agent(created=True)

    def reset_agent(self, created=False):
        if not created:
            logging.debug("\rresetting agent\n")
        self.agent = Agent(device=self.device,
                           state_size=self.environment.state_size(),
                           action_size=self.environment.action_size(),
                           layers=self.hp["layers"],
                           buffer_size=self.hp["buffer_size"],
                           batch_size=self.hp["batch_size"],
                           batch_frequency=self.hp["batch_frequency"],
                           epsilon_start=self.hp["epsilon_start"],
                           epsilon_end=self.hp["epsilon_end"],
                           epsilon_factor=self.hp["epsilon_factor"])
        return self.agent

    def create_environment(self, graphics=False):
        logging.info("\rcreating environment\n")
        self.environment = Environment(world=self.world, graphics=graphics)
        return self.reset_environment(created=True)

    def reset_environment(self, created=False):
        if not created:
            logging.debug("\rresetting environment\n")
        return self.environment.reset()

    def train(self, episodes=None, steps_per_episode=None, filename=None):
        # parameters
        episodes = self.hp["episodes"] if episodes is None else episodes
        steps_per_episode = self.hp["steps_per_episode"] if steps_per_episode is None else steps_per_episode

        # verify environment is created and reset
        self.create_environment() if self.environment is None else self.reset_environment()

        # verify agent is created and reset
        self.create_agent() if self.agent is None else self.reset_agent()

        self.log_hp(self.hp)

        scores_total = deque()             # scores over all episodes
        scores_window = deque(maxlen=100)  # scores over last 100 episodes
        average_scores = deque()           # all average scores over 100 episodes
        average_score = None
        episode_counter = 0

        logging.info("\rstart training:\n")

        for episode in range(1, episodes+1):
            episode_counter = episode
            state = self.reset_environment()  # get initial state
            score = 0  # episode score

            for step in range(steps_per_episode):
                action = self.agent.act(state=state)
                environment_change = self.environment.action(action)
                reward = environment_change["reward"]
                next_state = environment_change["next_state"]
                done = environment_change["done"]
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break  # break for steps iteration

            self.agent.update_epsilon()
            scores_total.append(score)   # save most recent score
            scores_window.append(score)  # save most recent score

            average_score = numpy.average(scores_window)  # average score over last 100 episodes
            average_scores.append(average_score)          # save most revent average score
            if not episode % 100 == 0:
                logging.info("\rEpisode {:5d} with Average Score: {:4.2f}".format(episode, average_score))
            elif episode % 100 == 0 and average_score < 13:
                logging.info("\rEpisode {:5d} with Average Score: {:4.2f}\n".format(episode, average_score))
            elif episodes % 100 == 0 and average_score >= 13:
                logging.info("\rSolved at Episode {:5d} with Average Score: {:4.2f}\n".format(episode, average_score))
                break  # break for episodes iteration

        if average_score is None:
            pass
        elif average_score < 13:
            logging.info("\rENVIRONMENT NOT SOLVED at Episode {:5d} with Average Sort: {:4.2f}\n"
                         .format(episode_counter, average_score))
        elif average_score >= 13:
            logging.info("\rENVIRONMENT SOLVED at Episode {:5d} with Average Sort: {:4.2f}\n"
                         .format(episode_counter, average_score))

        if filename is not None and self.agent is not None:
            self.agent.save()

        return list(scores_total), list(average_scores)

    def tune(self):
        for hp_key, hpr_key in zip(self.hp.keys(), self.hpr.keys()):
            if not hp_key == hpr_key:
                logging.error("\rINVALID HYPERPARAMETERS FOR TUNING\n")
                exit()

        self.log_hpr(self.hpr)

        best_score = 0
        best_episode_counter = None
        best_hp = None

        hp_iterators = [iter(hpr) for hpr in self.hpr.values()]
        hp_combinations = itertools.product(*hp_iterators)

        for hp_combination in hp_combinations:
            self.hp = OrderedDict(zip(self.hp.keys(), hp_combination))
            current_scores, average_scores = self.train()

            current_episodes_counter = len(current_scores)
            current_score = numpy.average(current_scores[-100:])
            if best_episode_counter is None or current_episodes_counter < best_episode_counter:
                best_episode_counter = current_episodes_counter
                best_score = current_score
                best_hp = self.hp.copy()

        logging.info("\rBEST SOLUTION AFTER {:4d} WITH A SCORE OF {:4.2f}\n".format(best_episode_counter, best_score))
        self.log_hp(best_hp, line=False)

    def show(self):
        pass

    @staticmethod
    def log_hp(hp, line=True):
        if line:
            logging.info("\r--------------------------------------\n")
        for key, value in hp.items():
            logging.info("\r{}: {}\n".format(key, value))

    @staticmethod
    def log_hpr(hpr):
        logging.info("\r--------------------------------------\n")
        for key, hyperparameter_range in hpr.items():
            start = hyperparameter_range[0]
            stop = hyperparameter_range[-1]
            if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
                if len(hyperparameter_range) > 1:
                    step = (hyperparameter_range[-1] - hyperparameter_range[0]) / (len(hyperparameter_range) - 1)
                if len(hyperparameter_range) == 1:
                    step = 0
                logging.info("\r{}: from {} to {} with step {}\n".format(key, start, stop, step))
        logging.info("\r--------------------------------------\n")
        total_combinations = numpy.product([len(element) for element in hpr.values()])
        logging.info("\rTOTAL COMBINATIONS FOR HYPERPARAMETER TUNING: {}\n".format(total_combinations))
        logging.info("\r--------------------------------------\n")

    @staticmethod
    def plot(scores, filename=None):
        fig = pyplot.figure()
        pyplot.plot(numpy.arange(len(scores)), scores)
        pyplot.ylabel('Score')
        pyplot.xlabel('Episode')
        if filename is None:
            pyplot.show()
        if filename is not None:
            pyplot.savefig(filename)

    @staticmethod
    def inclusive_range(start, stop, step=None):
        if step is None:
            step = 1
        return list(range(start=start, stop=stop+1, step=step))


if __name__ == "__main__":
    Navigation().setup(mode=sys.argv[1], world="Banana")
