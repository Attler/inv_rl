import numpy as np

import gym
from gym.envs.registration import register

from utils.general import to_one_hot


class CirlWrapper(gym.Wrapper):
    """Wrapper that changes the rewards recieved to match the CIRL algorithm,
    e.g. the reward is penalized by it's l2 distance from the expected feature
    distribution of the expert policy.

    The enviroment should return a feature vector in the info dictionary.
    """

    def __init__(self, env: gym.Env, expert_features, traj_state = False):
        """
        env: gym.Env
            The gym environment to be wrapped.
            This needs to be a featured algorithm, as if wrapped byFeatureWrapper!

        expert_features: numpy array of shape (d,)
            where d is the feature size of the env
            The feature distribution we are trying to match
        """
        super(CirlWrapper, self).__init__(env)
        self.expert_features = expert_features
        self.d = env.feature_dimensionality()[0]

        assert self.expert_features.shape == (self.d,)

        self.feature_trajectory = np.empty((0,self.d))

        self.traj_state = traj_state
        self.obs_n = self.observation_space.n

        if self.traj_state:
            self.observation_space = gym.spaces.Box(0, 1,
                shape=(self.obs_n + self.d,),
                dtype=np.float32)

    def step(self, action):
        next_state, reward, terminated, info = self.env.step(action)
        f = info["features"]

        self.feature_trajectory = np.concatenate(
            (self.feature_trajectory, f), axis=0)

        if terminated:
            reward -= self.feature_dist()

        if self.traj_state:
            next_state = self.traj_state_vector(next_state)
        return next_state, reward, terminated, {}

    def traj_state_vector(self, state):
        return np.append(
            to_one_hot(state, self.obs_n),
            self.avg_traj()
        )

    def reset(self):
        self.feature_trajectory = np.empty((0,self.d))
        s = self.env.reset()
        if self.traj_state:
            s = self.traj_state_vector(s)
        return s

    def avg_traj(self):
        """ Returns ndarray of size (d,) corresponding to the average features
        across the current trajectory
        """
        if self.feature_trajectory.shape[0]:
            return self.feature_trajectory.mean(axis=0)
        else:
            return np.zeros(self.d,)

    def feature_dist(self, eta=.01):
        """ Returns the l2 distance of the average features for the current
        trajectory and the expert features.
        """
        dist_vec = self.avg_traj() - self.expert_features
        return eta * np.linalg.norm(dist_vec, ord=2)
