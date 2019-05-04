import numpy as np
from gym.envs.toy_text import discrete
from scipy.interpolate import Rbf
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
from matplotlib import cm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NULL = 4

class InterruptionEnv(discrete.DiscreteEnv):
    """


    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3, NULL=4).

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, random_start=False, Ua=-1, sigmaUa=1, beta=1): #DO NOT SET beta=0 (beta=0 is rational and beta=infty is random)

        self.shape = (9, 9)
        self.random_start = random_start
        self.Ua = Ua
        self.sigmaUa = sigmaUa
        self.beta = beta

        self.nS = np.prod(self.shape)
        self.nA = 5

        self.MAX_Y = self.shape[0]
        self.MAX_X = self.shape[1]

        self.Goal=(2,3) # (x,y) coordinates
        self.Stop=(4,5)
        self.Ask=(6,3)

        self.x = (self.Goal[0], self.Ask[0], self.Stop[0])
        self.y = (self.Goal[1], self.Ask[1], self.Stop[1])

        self.grid = self.calc_grid_reward()

        self.P = self.calc_reward_stop_probability()




        # Initial state distribution is uniform
        if self.random_start:
            isd = np.ones(self.nS) / self.nS
        else:
            isd = np.zeros(self.nS)
            start = np.ravel_multi_index((self.MAX_Y//2, self.MAX_X//2), self.shape)
            isd[start] = 1

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm

        super(InterruptionEnv, self).__init__(self.nS, self.nA, self.P, isd)


    def calc_reward_stop_probability(self):

        def is_done(s,is_pedagogic): #Is it ok to define a function inside another function
            ay, ax = np.unravel_index(s, self.shape)
            for i in range(0, len(self.x)):
                if (ax, ay) == self.Goal:
                    return True  # Has collected reward, stop.
                if (ax, ay) == self.Stop:
                    return True  # Has stoped
                if (ax, ay) == self.Ask and is_pedagogic == False:
                    return True
                if (ax, ay) == self.Ask and is_pedagogic == True: #If we are in a pedagogic trajectory, we do not want to make the ask a terminal state
                    return False
            return False  # keep going, non interesting state

        it = np.nditer(self.grid, flags=['multi_index'])

        P = {}
        Q = {}

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            Q[s] = {a: [] for a in range(self.nA)}

            reward = self.grid[y][x]

            if is_done(s,is_pedagogic=False): #It is not clear if I can use False like this
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                P[s][NULL] = [(1.0, s, reward, True)]
            if is_done(s,is_pedagogic=True): #It is not clear if I can use True like this
                Q[s][UP] = [(1.0, s, reward, True)]
                Q[s][RIGHT] = [(1.0, s, reward, True)]
                Q[s][DOWN] = [(1.0, s, reward, True)]
                Q[s][LEFT] = [(1.0, s, reward, True)]
                Q[s][NULL] = [(1.0, s, reward, True)]

            else:  # Not a terminal state
                ns_up = s if y == 0 else s - self.MAX_X
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                ns_left = s if x == 0 else s - 1
                ns_null = s

                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up,False))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right,False))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down,False))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left,False))]
                P[s][NULL] = [(1.0, s, reward, is_done(ns_null,False))]

                Q[s][UP] = [(1.0, ns_up, reward, is_done(ns_up,True))]
                Q[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right,True))]
                Q[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down,True))]
                Q[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left,True))]
                Q[s][NULL] = [(1.0, s, reward, is_done(ns_null,True))]
            it.iternext()
        return P, Q

    def calc_grid_reward(self):
        r=np.random.normal(self.Ua, self.sigmaUa) #This way the reward will be consistent over the episodo
        # x,y -> coord, d -> value, Ua= reward
        if self.Ua >= 0:  # rational interruptor will not stop him
            d=  (r, np.random.choice((r, 0), p=(
                1 / (1 + np.exp(-self.Ua / self.beta)), 1 - 1 / (1 + np.exp(-self.Ua / self.beta)))), 0)
        if self.Ua < 0:  # rational interruptor will stop him
            d= (r, np.random.choice((0, r), p=(
                1 / (1 + np.exp(self.Ua / self.beta)), 1 - 1 / (1 + np.exp(self.Ua / self.beta)))), 0)

        d = (np.random.normal(self.Ua, self.sigmaUa), 0, 0)
        grid = np.zeros(self.shape)
        for i in range(0, len(self.x)):
            grid[self.y[i]][self.x[i]] = d[i]
        return grid

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

        return np.eye(self.nS)[state, :]

    def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.reward = self.calc_grid_reward()   #Is this ok?
        self.P, self.Q = self.calc_reward_stop_probability()
        return self.s