
"""Module for maximum entropy inverse reinforcement learning."""

import gym
import numpy as np

from wrappers import CirlWrapper

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from utils.callbacks import ActionTrajectoryCallback

class RLSearchCirlAlg(BaseCirlAlg):
    """
    Train an RL algorithm to search for pedalogical trajectories.
    """

    def __init__(self, env: gym.Env, exptected_features):
        super(RLSearchCIRL, self).__init__(env, exptected_features, traj_state=True)

    def DQN_agent(self, limit=None):
        """ Trains a DQN agent to find the most pedalogical trajectory,
        as defined by equation (1) in the CIRL paper.
        This trajectory maximizes reward while being close
        to the expected feature values for the expert policy.

        Agent training code is taken from
        https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py
        """

        n_actions = self.env.action_space.n
        # Next, we build a very simple model.
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(n_actions))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory,
                        nb_steps_warmup=500, target_model_update=1e-2,
                        policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        dqn.fit(self.env, nb_steps=50000, verbose=1)

        return dqn

    def cirl_trajectory(self, alg="dqn"):

        if alg == "dqn":
            agent = self.DQN_agent()

        cb = ActionTrajectoryCallback()

        agent.test(self.env, nb_episodes=1, verbose=1, visualize=False, callbacks=[cb])

        return cb.actions
