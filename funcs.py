import numpy as np


def run_bandits(a, t, mu, sigma):

    """This is a simple implementation of the n-armed bandits problem as described in Sutton and Barto's
    Reinforcement Learning Book. This problem introduces some basic intuition about stationary reinforcement learning problems.

    a .. number of bandits
    t .. iterations
    mu .. mean of the reward distribution
    sigma .. std of the reward distribution

    """

    epsilons = [0.05, 0.1, 0.2, 0.3]
    q_a = np.random.normal(mu, sigma, (10, 1))
    epsilon_dict = {}

    if type(a) != int:
        raise ValueError("the number of bandits (a) must be specified as an integer.")

    for epsilon in epsilons:
        Q_a = np.zeros((10, 1))
        N_a = np.zeros((10, 1))
        for i in range(t):
            # e-greedy action selection:
            is_greedy = np.random.choice([0, 1], p=[epsilon, 1 - epsilon])
            if is_greedy == 1:
                A = np.argmax(Q_a)
            if is_greedy == 0:
                A = np.random.randint(1, a)
            # pull bandit:
            R = np.random.normal(q_a[A], 1)
            # update rules:
            N_a[A] += 1
            Q_a[A] += (1 / N_a[A]) * (R - Q_a[A])
        epsilon_dict[epsilon] = N_a, Q_a
        return epsilon_dict


def report_bandits(a, epsilon_dict):
    """This is a reporting function for the bandit problem.

    a .. number of bandits
    episolon_dict .. results per epsilon

    """
    dict_mean_reward = {}
    for k, v in epsilon_dict.items():
        dict_mean_reward[k] = (np.dot(v[0].T, v[1]) / a)[0][
            0
        ]  # grabbing just the number from the array
    best_epsilon = max(
        dict_mean_reward, key=dict_mean_reward.get
    )  # getting the key of the max value
    print(
        "The epsilon value that maximises the average reward is: {0}. \nThe avergae reward is: {1}".format(
            best_epsilon, dict_mean_reward[best_epsilon]
        )
    )
    return best_epsilon
