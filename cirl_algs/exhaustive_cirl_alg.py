import gym
import numpy as np

from cirl_algs.base_cirl_alg import BaseCirlAlg

from math import inf
from tqdm import tqdm

class ExhaustiveCirlAlg(BaseCirlAlg):
    """
    Exhausitvely search all trajectories, find the best one by the CIRL
    objective formula.
    """

    def __init__(self, env: gym.Env, exptected_features):
        super(ExhaustiveCirlAlg, self).__init__(env, exptected_features)

    def cirl_trajectory(self, limit=None):
        """ Finds the most pedalogical trajectory, as defined by equation (1)
        in the CIRL paper.  This trajectory maximizes reward while being close
        to the expected feature values for the expert policy.
        """
        actions = np.arange(self.env.action_space.n)
        trajs = self._cartesian_product(*[actions for _ in range(10)])

        if limit is not None:
            trajs = trajs[:limit]

        rewards = np.zeros(len(trajs))
        for i, t in enumerate(tqdm(trajs)):
            rewards[i] = self.cirl_reward(t)

        print(np.argmax(rewards))

        return trajs[np.argmax(rewards)]

    def cirl_reward(self, traj : np.ndarray) -> float:
        self.env.reset()

        reward: float
        for a in traj:
            _ , reward, terminated, _ = self.env.step(a)
            if terminated:
                return reward

        return reward - self.env.feature_dist()

    def _cartesian_product(self,*arrays):
        """
        From https://stackoverflow.com/a/11146645
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
