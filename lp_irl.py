import numpy as np
from scipy.optimize import linprog

def lp_irl(trans_probs, policy, gamma=0.2, l1=1.5, Rmax=5.0):
    """
    Solve Linear programming for Inverse Reinforcement Learning
    """
    n_states, n_actions, _ = trans_probs.shape
    A = set(range(n_actions))
    tp = np.transpose(trans_probs, (1, 0, 2))
    ones_s = np.ones(n_states)
    eye_ss = np.eye(n_states)
    zero_s = np.zeros(n_states)
    zero_ss = np.zeros((n_states, n_states))
    T = lambda a, s: np.dot(tp[policy[s], s] - tp[a, s], np.linalg.inv(eye_ss - gamma * tp[policy[s]]))

    c = -np.r_[zero_s, ones_s, -l1 * ones_s]
    zero_stack = np.zeros((n_states * (n_actions - 1), n_states))
    T_stack = np.vstack([-T(a, s) for s in range(n_states) for a in A - {policy[s]}])
    I_stack = np.vstack([np.eye(1, n_states, s) for s in range(n_states) for a in A - {policy[s]}])

    A_ub = np.bmat([[T_stack, I_stack, zero_stack],    # -TR <= t
                    [T_stack, zero_stack, zero_stack], # -TR <= 0
                    [-eye_ss, zero_ss, -eye_ss],  # -R <= u
                    [eye_ss, zero_ss, -eye_ss],   # R <= u
                    [-eye_ss, zero_ss, zero_ss],  # -R <= Rmax
                    [eye_ss, zero_ss, zero_ss]])  # R <= Rmax
    b = np.vstack([np.zeros((n_states * (n_actions-1) * 2 + 2 * n_states, 1)),
                   Rmax * np.ones((2 * n_states, 1))])
    results = linprog(c, A_ub, b)
    r = np.asarray(results["x"][:n_states], dtype=np.double)

    return r.reshape((n_states,))

if __name__ == '__main__':
    from envs import gridworld
    from value_iteration import *

    def trans_mat(env):
        return np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)])
    
    grid = gridworld.GridworldEnv()
    U = value_iteration(grid)
    pi = best_policy(grid, U)

    res = lp_irl(trans_mat(grid), pi)
    print res

    import matplotlib.pyplot as plt
    def to_mat(res, shape=(4, 4)):
        dst = np.zeros(shape)
        for i, v in enumerate(res):
            dst[i / 4, i % 4] = v
        return dst

    plt.matshow(to_mat(res))
    plt.show()
