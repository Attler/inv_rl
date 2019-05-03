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

class RbfGridworldEnv(discrete.DiscreteEnv):
    """


    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3, NULL=4).

    """

    metadata = {'render.modes': ['human', 'ansi']}
    def is_done(s,self):
        ay, ax = np.unravel_index(s, self.shape)
        for i in range(0,len(self.x)):
            if (ax,ay)==self.Goal:
                return True #Has collected reward, stop.
            if (ax,ay)==self.Stop:
                return True #Has stoped
            if (ax,ay)==self.Ask:
                if self.Ua >= 0: #rational interruptor will not stop him
                    return np.random.choice((False, True), p=(1/(1+np.exp(-self.Ua/self.beta)), 1-1/(1+np.exp(-self.Ua/self.beta)) ) )
                if self.Ua < 0:  #rational interruptor will stop him
                    return np.random.choice((True, False), p=(1/(1+np.exp(self.Ua/self.beta)), 1-1/(1+np.exp(self.Ua/self.beta)) ) )
        return False #keep going, non interesting state

    def calc_reward_stop_probability(self):
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            reward = calc_grid_reward(self)

            if is_done(s,self):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                P[s][NULL] = [(1.0, s, reward, True)]

            else:         # Not a terminal state
                ns_up = s if y == 0 else s - self.MAX_X
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                ns_left = s if x == 0 else s - 1
                ns_null = s

                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up,self))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right,self))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down,self))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left,self))]
                P[s][NULL] = [(1.0, s, reward, is_done(ns_null,self))]

                it.iternext()
        return P
    
    def calc_grid_reward(self):
            # x,y -> coord, d -> value, Ua= reward
            d = (np.random.normal(Ua,sigmaUa), 0, 0)
            self.grid = np.zeros(self.shape)
            for i in range(0,len(self.x)):
                self.grid[self.y[i]][self.x[i]]=d[i]
            return self.grid

    def __init__(self, random_start=False, Ua=1, sigmaUa=1, beta=1): #DO NOT SET beta=0 (beta=0 is rational and beta=infty is random)

        self.shape = (9, 9)
        self.random_start = random_start

        nS = np.prod(self.shape)
        nA = 5

        self.MAX_Y = self.shape[0]
        self.MAX_X = self.shape[1]


        P = {}

        Goal=(2,3) # (x,y) coordinates
        Stop=(4,5)
        Ask=(6,3)

        x = (Goal[0], Ask[0], Stop[0])
        y = (Goal[1], Ask[1], Stop[1])

        it = np.nditer(self.grid, flags=['multi_index'])
        
        P = calc_reward_stop_probability(self)




        # Initial state distribution is uniform
        if self.random_start:
            isd = np.ones(nS) / nS
        else:
            isd = np.zeros(nS)
            start = np.ravel_multi_index((self.MAX_Y//2, self.MAX_X//2), self.shape)
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

        features = []
        for i in range(len(self.centers[0])):
            dist = np.linalg.norm([(x, y), (self.centers[0][i], self.centers[1][i])])
            features.append(dist)
        return features

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.reward = calc_grid_reward(self)   #Is this ok?
        self.P = calc_reward_stop_probability(self)
        return self.s