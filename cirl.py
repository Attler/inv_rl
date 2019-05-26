from max_ent_irl import *
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def dist_feature_matrix(env):
    return np.array([env.gen_features(s) for s in range(env.nS)])


def generate_demos(env, trans_probs, U, n_trajs=100, len_traj=5):
    trajs = []
    for _ in range(n_trajs):
        episode = []
        env.reset()
        for i in range(len_traj):
            cur_s = env.s
            policy=best_policy(trans_probs, U)
            state, reward, done, _ = env.step(policy[cur_s])
            episode.append((cur_s, policy[cur_s], state))
            if done:
                for _ in range(i + 1, len_traj):
                    episode.append((state, 5, state))
                break
        trajs.append(episode)
    return trajs


def generate_pedagogic(expert_trajs, env, len_traj=10):
    
    # Generate all possible trajectories
    possible_trajs = []

    all_possible_action_combinations = list(itertools.combinations_with_replacement(np.arange(env.nA), len_traj))

    random.shuffle(all_possible_action_combinations)

    for combination in all_possible_action_combinations:
        state0 = env.reset()
        traj=[]
        for action in combination:

            # P[state][action] = [(prob, n_state, reward, is_done)]

            # state, action, n_state, reward.
            traj.append((state0, action, env.P[state0][action][0][1], env.P[state0][action][0][2]))
            state0 = env.P[state0][action][0][1] #Update the current state to the new one
        possible_trajs.append(traj)


    # Expert sum of features
    expert_sum_of_features = np.zeros(len(env.gen_features(0)))

    for traj in expert_trajs:
        for transition in traj:
            expert_sum_of_features += env.gen_features(transition[0])
        expert_sum_of_features += env.gen_features(traj[-1][2])

    expert_sum_of_features = expert_sum_of_features / len(expert_trajs)

    # Select the best trajectories
    maximum = 0
    trajs_goodness = {}  # How good is the trajectory according to

    eta = 0.0001  # Eta in the formula from CIRL

    for i, traj in enumerate(possible_trajs):
        sum_of_features = np.zeros(len(env.gen_features(0)))
        traj_reward = 0

        for transition in traj:
            sum_of_features += env.gen_features(transition[0])  # setting features = coordinates
            traj_reward += transition[3]

        sum_of_features += env.gen_features(traj[-1][2])

        trajs_goodness[i] = traj_reward - eta * np.linalg.norm(sum_of_features - expert_sum_of_features)

    pedagogical_trajs = []

    best_k = sorted(trajs_goodness, key=trajs_goodness.get, reverse=True)[:3]

    print(best_k)

    for i, traj in enumerate(possible_trajs):
        if i in best_k:
            pedagogical_trajs.append(traj)
            print(traj)

    return pedagogical_trajs


if __name__ == '__main__':
    from envs import rbfgridworld

    grid_shape = (9, 9)
    grid = rbfgridworld.RbfGridworldEnv(grid_shape)

    trans_probs, reward = trans_mat(grid)
    U = value_iteration(trans_probs, reward, gamma=0.2)
    pi = best_policy(trans_probs, U)

    # Trajectories
    n_traj = 8
    expert_trajs = generate_demos(grid, trans_probs, U, len_traj=n_traj, n_trajs=3)
    pedagogic_trajs = generate_pedagogic(expert_trajs, grid, len_traj=n_traj)

    # Learning
    res_irl  = max_ent_irl(feature_matrix(grid), trans_probs, expert_trajs, alpha=0.4, gamma=0.999)
    res_cirl = max_ent_irl(feature_matrix(grid), trans_probs, pedagogic_trajs, alpha=0.4, gamma=0.999)
    print("IRL:", res_irl)
    print("CIRL:", res_cirl)
    ##############################################################################

    ax1 = plt.subplot(1,3,1)
    ax1.set_title("GroundTruth Rewards")
    plt.matshow(grid.grid, cmap=cm.Blues_r, fignum=False)

    ######IRL
    ax2 = plt.subplot(1,3,2)
    ax2.set_title("IRL Rewards")
    plt.matshow(np.reshape(res_irl, grid.shape), cmap=cm.Blues_r, fignum=False)

    counts = np.zeros(grid_shape)
    for traj in expert_trajs:
        for step in traj:
            y, x = np.unravel_index(step[0], grid.shape)
            counts[x,y]+=1
    for x in range(9):
        for y in range(9):
            if counts[x, y] != 0:
                plt.text(x, y, int(counts[x,y]), fontsize=8)


    ######CIRL
    ax1 = plt.subplot(1, 3, 3)
    ax1.set_title("CIRL Rewards")
    plt.matshow(np.reshape(res_cirl, grid.shape), cmap=cm.Blues_r, fignum=False)

    counts = np.zeros((9,9))
    for traj in pedagogic_trajs:
        for step in traj:
            y, x = np.unravel_index(step[0], grid.shape)
            counts[x,y]+=1
    for x in range(9):
        for y in range(9):
            if counts[x, y] != 0:
                plt.text(x, y, int(counts[x,y]), fontsize=8)

    plt.show()
