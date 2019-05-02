"""Module for feature wrappers providing features for different environments."""

from abc import abstractmethod
import functools
from typing import Union, Tuple

import gym
import numpy as np

from utils.general import to_one_hot

class FeatureWrapper(gym.Wrapper):
    """Wrapper that adds features to the info dictionary in the step function.

    Generally, each environment needs its own feature wrapper."""

    def __init__(self, env: gym.Env):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment to be wrapped.
        """
        super(FeatureWrapper, self).__init__(env)
        self.current_state = None

        self.d = env.feature_dimensionality()[0]
        self.feature_trajectory = np.empty((0,self.d))

    def reset(self, **kwargs):  # pylint: disable=method-hidden, R0801
        """ Reset environment and return initial state.
        No changes to base class reset function."""
        self.current_state = self.env.reset()
        self.feature_trajectory = np.empty((0,self.d))
        return self.current_state

    def step(self, action: Union[np.ndarray, int, float]
             ) -> Tuple[Union[np.ndarray, float, int], float, bool, dict]:
        """

        Parameters
        ----------
        action: Union[np.ndarray, int, float]

        Returns
        -------
        Tuple[Union[np.ndarray, float, int], float, bool, dict]
            Tuple with values for (state, reward, done, info).
            Normal return values of any gym step function.
            A field 'features' is added to the returned info dict.
            The value of 'features' is a np.ndarray of shape (1, d)
            where d is the dimensionality of the feature space.
        """
        # pylint: disable=E0202

        # execute action:
        next_state, reward, terminated, info = self.env.step(action)

        info['features'] = self.features(self.current_state, action,
                                         next_state)

        self.feature_trajectory = np.concatenate(
            (self.feature_trajectory, info['features']), axis=0)

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    def avg_traj(self):
        """ Returns ndarray of size (d,) corresponding to the average features
        across the current trajectory
        """
        if self.feature_trajectory.shape[0]:
            return self.feature_trajectory.mean(axis=0)
        else:
            return np.zeros(self.d,)

    @abstractmethod
    def features(self, current_state: Union[np.ndarray, int, float],
                 action: Union[np.ndarray, int, float],
                 next_state: Union[np.ndarray, int, float]) -> np.ndarray:
        """Return features for a single state or a state-action pair or a
        state-action-next_state triplet. To be saved in step method's info dictionary.

        Parameters
        ----------
        current_state: Union[np.ndarray, int, float]
            The current state. Can be None if used reward function has properties
            action_in_domain == False and next_state_in_domain == False and if
            next_state is not None. In that case the features are calculated for
            the next state and used for the reward function R(s) - the reward for
            reaching next_state.
        action: Union[np.ndarray, int, float]
            A single action. Has to be given if used reward function has property
            action_in_domain == True.
        next_state: Union[np.ndarray, int, float]
            The next state. Has to be given if used reward function has property
            next_state_in_domain == True.

        Returns
        -------
        np.ndarray
            The features in a numpy array of shape (1, d), where d is the
            dimensionality of the feature space (see :meth:`.feature_dimensionality`).
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_dimensionality(self) -> tuple:
        """Get the dimensionality of the feature space."""
        raise NotImplementedError()

    @abstractmethod
    def feature_range(self) -> np.ndarray:
        """Get minimum and maximum values of all d features, where d is the
        dimensionality of the feature space (see :meth:`.feature_dimensionality`)

        Returns
        -------
        np.ndarray
            The minimum and maximum values in an array of shape (2, d).
            First row corresponds to minimum values and second row to maximum values.
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_array(self) -> np.ndarray:
        """ Get features for the entire domain as an array.
        Has to be overwritten in each feature wrapper.
        Wrappers for large environments will not implement this method.

        Returns
        -------
        np.ndarray
            The features for the entire domain as an array.
            Shape: (domain_size, d).
        """
        raise NotImplementedError()


class GridPosFeatureWrapper(FeatureWrapper):
    """Feature wrapper that was ad hoc written for the FrozenLake env.

    Would also work to get one-hot features for any other discrete env
    such that feature-based IRL algorithms can be used in a tabular setting.
    """

    def features(self, current_state: None, action: None,
                 next_state: int) -> np.ndarray:
        """Return features to be saved in step method's info dictionary.
        One-hot encoding the next state.

        Parameters
        ----------
        current_state: None
        action: None
        next_state: int
            The next state.

        Returns
        -------
        np.ndarray
            The features in a numpy array.
        """
        assert next_state is not None
        if isinstance(next_state, (int, np.int64, np.ndarray)):
            return to_one_hot(next_state, self.env.observation_space.n)
        else:
            raise NotImplementedError()

    def feature_dimensionality(self) -> Tuple:
        """Return dimension of the one-hot vectors used as features."""
        return (self.env.observation_space.n, )

    def feature_range(self):
        """Get maximum and minimum values of all k features.

        Returns:
        `np.ndarray` of shape (2, k) w/ max in 1st and min in 2nd row.
        """
        ranges = np.zeros((2, self.feature_dimensionality()[0]))
        ranges[1, :] = 1.0
        return ranges

    def feature_array(self):
        """Returns feature array for FrozenLake. Each state in the domain
        corresponds to a one_hot vector. Features of all states together
        are the identity matrix."""
        return np.eye(self.env.observation_space.n)
