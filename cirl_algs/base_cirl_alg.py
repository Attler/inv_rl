import gym
from wrappers import CirlWrapper

class BaseCirlAlg():
    def __init__(self, env: gym.Env, exptected_features, traj_state=False):
        self.env = CirlWrapper(env, exptected_features, traj_state = traj_state)

    def cirl_trajectory(self):
        raise NotImplimentedError()
