import numpy as np
from gym.envs.toy_text import discrete
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import cm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NULL = 4

class RbfGridworldEnv(discrete.DiscreteEnv):
    """
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3, NULL=4).
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=(9,9), random_start=False):

        def is_done(s):
            ay, ax = np.unravel_index(s, self.shape)
            r = self.grid[ay][ax]
            if r == -1 or r == 1:
                return True
            return False

        def is_done(s):
            return False

        self.shape = shape
        self.random_start = random_start

        nS = np.prod(self.shape)
        nA = 4

        self.MAX_Y = self.shape[0]
        self.MAX_X = self.shape[1]


        P = {}

        theta = np.array((1, 1, -1))
        centers = np.array([[2, 3],
                            [6, 3],
                            [4, 6]], dtype=np.float_)

        grid_index = np.asarray([[y, x] for x, y in np.ndindex(self.shape[0], self.shape[1])], dtype=np.float_)

        self.state_features = np.exp(-(cdist(grid_index, centers) / 1.6) ** 2)

        self.grid = np.dot(self.state_features, theta).reshape(self.shape[0], self.shape[1])




        it = np.nditer(self.grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            reward = self.grid[y][x]

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)] #[prob, next state, reward, is_done]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                P[s][NULL] = [(1.0, s, reward, True)]
            # Not a terminal state

            else:
                ns_up = s if y == 0 else s - self.MAX_X
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                ns_left = s if x == 0 else s - 1

                P[s][UP] = [(1.0, ns_up, reward, False)]
                P[s][RIGHT] = [(1.0, ns_right, reward, False)]
                P[s][DOWN] = [(1.0, ns_down, reward, False)]
                P[s][LEFT] = [(1.0, ns_left, reward, False)]
                P[s][NULL] = [(1.0, s, reward, False)]

            it.iternext()

        # Initial state distribution is uniform
        if self.random_start:
            isd = np.ones(nS) / nS
        else:
            isd = np.zeros(nS)
            start = np.ravel_multi_index((self.MAX_Y//2, self.MAX_X//2), self.shape)
            #start = np.ravel_multi_index((2, 4), self.shape)
            isd[start] = 1

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(RbfGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        if close:
            return

        if mode == 'rgb_array':
            xi = np.linspace(0, self.shape[0]-1, self.shape[0])
            yi = np.linspace(0, self.shape[1]-1, self.shape[1])
            XI, YI = np.meshgrid(xi, yi)
            plt.subplot(1, 1, 1)
            plt.pcolor(XI, YI, self.grid, cmap=cm.Blues_r)
            plt.colorbar()
            ax, ay = np.unravel_index(self.s, self.shape)
            plt.scatter(ax+0.5, ay+0.5, c='black', marker='x')
            plt.show()

    def gen_features(self, state):

        y, x = np.unravel_index(state, self.shape)

        features = [np.linalg.norm([np.array((x, y)) - (0, 0)]),
                    np.linalg.norm((np.array((x, y)) - np.array((self.MAX_X, self.MAX_Y))))]
        return features





