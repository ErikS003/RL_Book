from k_bandits import QLearningAgent, train
import numpy as np



class NonstationaryKArmedBandit:
    def __init__(self, k=10, drift = 0.01):
        self.k = k
        # Initialize the true action values (mean reward) for each arm
        self.drift = drift

        self.q_true = np.random.normal(0, 1, k)

    def step(self, action):
        # Reward is sampled from a normal distribution centered at the true value of the action
        reward = np.random.normal(self.q_true[action], 1)
        # Each action value takes a random walk

        self.q_true = self.q_true + np.random.normal(0, self.drift, self.k)

        return reward
    
def train(agent, bandit, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        action = agent.choose_action()
        reward = bandit.step(action)
        agent.update_q(action, reward)
        rewards[step] = reward
    return rewards    

import matplotlib.pyplot as plt
k = 20
steps = 10000
drift = 0.01  # std of the noise/drift i.e q_true = q_true + N(0,noise)

nonstationary_bandit = NonstationaryKArmedBandit(k, drift)
epsilon_greedy_agent = QLearningAgent(k, alpha=1/steps, epsilon=0.1)

# Train the agent in the nonstationary environment
rewards = train(epsilon_greedy_agent, nonstationary_bandit, steps)

# Plot cumulative reward
plt.plot(np.cumsum(rewards), label='Epsilon-Greedy with Nonstationary Bandit')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title(f'{k}-Armed Bandit Problem with Nonstationarity (Drift = {drift})')
plt.legend()
plt.show()




# what can happen in nonstationarity:
# we find a good exploit i.e argmax(Q_t(a)) but this Q* drifts and gets overtaken by other Qs.
# we can't learn the optimal policy because the optimal policy is changing.