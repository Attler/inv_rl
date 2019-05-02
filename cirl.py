import numpy as np
from value_iteration import *
import itertools


'''''' #From irl-me
def expected_svf(trans_probs, trajs, policy): #state value function
    n_states, n_actions, _ = trans_probs.shape
    n_t = len(trajs[0])
    mu = np.zeros((n_states, n_t))
    for traj in trajs:
        mu[traj[0][0], 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)
    for t in range(1, n_t):
        for s in range(n_states):
            mu[s, t] = sum([mu[pre_s, t - 1] * trans_probs[pre_s, np.argmax(policy[pre_s]), s] for pre_s in range(n_states)])
    return np.sum(mu, 1)

def max_ent_irl(feature_matrix, trans_probs, trajs, gamma=0.9, n_epoch=20, alpha=0.5):
    n_states, d_states = feature_matrix.shape
    _, n_actions, _ = trans_probs.shape

    feature_exp = np.zeros((d_states))
    for episode in trajs:
        for step in episode:
            feature_exp += feature_matrix[step[0], :]
    feature_exp = feature_exp / len(trajs)

    theta = np.random.uniform(size=(d_states,))
    for _ in range(n_epoch):
        r = feature_matrix.dot(theta)
        v = value_iteration(trans_probs, r, gamma)
        pi = best_policy(trans_probs, v)
        exp_svf = expected_svf(trans_probs, trajs, pi)
        grad = feature_exp - feature_matrix.T.dot(exp_svf)
        theta += alpha * grad

    return feature_matrix.dot(theta)


def feature_matrix(env):
    return np.eye(env.nS)

def generate_demons(env, policy, n_trajs=100, len_traj=5, epsilon=1e-2):
    trajs = []
    for _ in range(n_trajs):
        episode = []
        env.reset()
        for i in range(len_traj):
            cur_s = env.s
            action = np.random.choice(np.arange(env.nA), p=policy[cur_s])
            state, reward, done, _ = env.step(action)
            episode.append((cur_s, action, state))
            if done:
                for _ in range(i + 1, len_traj):
                    episode.append((state, 0, state))
                break
        trajs.append(episode)
    return trajs


def generate_pedagogic(expert_trajs, env, len_traj=10):

    #Generate all possible trajectories
    possible_trajs = []

    all_possible_action_combinations = itertools.combinations_with_replacement(np.arange(env.nA), len_traj)
    for combination in all_possible_action_combinations:
        state0 = env.reset()
        traj=[]
        for action in combination:

            # P[state][action] = [(prob, n_state, reward, is_done)]

            # state, action, n_state, reward
            traj.append((state0, action, env.P[state0][action][0][1], env.P[state0][action][0][2]))
            state0 = env.P[state0][action][0][1] #Update the current state to the new one
        possible_trajs.append(traj)


    #Expert sum of features

    expert_sum_of_features = np.zeros(len(env.gen_features(0)))

    for traj in expert_trajs:

        for transition in traj:
            expert_sum_of_features += env.gen_features(transition[0]) #setting features = coordinates
        expert_sum_of_features += env.gen_features(traj[-1][2])

    expert_sum_of_features = expert_sum_of_features / len(expert_trajs)

    #Select the best trajectories
    maximum = 0
    trajs_goodness={} #How good is the trajectory according to

    #print(possible_trajs[45])

    for i, traj in enumerate(possible_trajs):

        sum_of_features = np.zeros(len(env.gen_features(0)))
        traj_reward=0
        eta=0.1 # Eta in the formula from CIRL

        for transition in traj:
            sum_of_features += env.gen_features(transition[0]) #setting features = coordinates
            traj_reward += transition[3]

        sum_of_features += env.gen_features(traj[-1][2])

        trajs_goodness[i] = traj_reward - eta * np.linalg.norm(sum_of_features - expert_sum_of_features)



    best_score=max(trajs_goodness.values())
    #print(trajs_goodness.values())

    eps = 0.1 # in order to select all the trajectories that are near the maximum

    pedagogical_trajs=[]

    best_k = sorted(trajs_goodness, key=trajs_goodness.get, reverse=True)[:10]

    for i, traj in enumerate(possible_trajs):
        if i in best_k:
            pedagogical_trajs.append(traj)

    return pedagogical_trajs


        

if __name__ == '__main__':
    from envs import rbfgridworld
    grid = rbfgridworld.RbfGridworldEnv()


    from envs import gridworld
    #grid = gridworld.GridworldEnv(shape=(5,5))

    trans_probs, reward = trans_mat(grid)
    U = value_iteration(trans_probs, reward)
    pi = best_policy(trans_probs, U)

    n_traj = 16
    expert_trajs = generate_demons(grid, pi, len_traj=n_traj)

    pedagogic_trajs = generate_pedagogic(expert_trajs, grid, len_traj=n_traj)

    res = max_ent_irl(feature_matrix(grid), trans_probs, pedagogic_trajs)
    print(res)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.matshow(grid.grid, 1, cmap=cm.Blues_r)

    def to_mat(res, shape):
        dst = np.zeros(shape)
        for i, v in enumerate(res):
            dst[i // shape[1], i % shape[1]] = v
        return dst

    plt.matshow(to_mat(res, grid.shape),2, cmap=cm.Blues_r)

    xs = []
    ys = []

    for traj in pedagogic_trajs:
        for step in traj:
            y, x = np.unravel_index(step[0], grid.shape)
            xs.append(x)
            ys.append(y)
        plt.scatter(xs, ys, marker='X', color='Black')

    plt.show()
