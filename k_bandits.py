import numpy as np
import matplotlib.pyplot as plt

# Set up the k-armed bandit problem
class KArmedBandit:
    def __init__(self, k=10):
        self.k = k
        # Initialize the true action values (mean reward) for each arm
        self.q_true = np.random.normal(0, 1, k)

    def step(self, action):
        # Reward is sampled from a normal distribution centered at the true value of the action
        reward = np.random.normal(self.q_true[action], 1)
        return reward

# Q-learning agent
class QLearningAgent:
    def __init__(self, k=10, alpha=0.1, epsilon=0.1):
        self.k = k
        self.alpha = alpha  # Learning rate/step size
        self.epsilon = epsilon  # Exploration rate (0 for greedy method)
        self.q_estimates = np.zeros(k)  # Estimated action values
        self.action_counts = np.zeros(k)  # Counts for each action

    def choose_action(self):
        if self.epsilon > 0 and np.random.rand() < self.epsilon:
            return np.random.choice(self.k)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def update_q(self, action, reward):
        # Incremental Q-learning update
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
        self.action_counts[action] += 1

# Training function for an agent
def train(agent, bandit, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        action = agent.choose_action()
        reward = bandit.step(action)
        agent.update_q(action, reward)
        rewards[step] = reward
    return rewards

k = 20
n = 10000

bandit = KArmedBandit(k)

# Epsilon-greedy (e.g., epsilon = 0.1)
epsilon_greedy_agent = QLearningAgent(k, alpha=1/n, epsilon=0.1)
epsilon_greedy_rewards = train(epsilon_greedy_agent, bandit, n)

#Purely greedy (epsilon = 0)
greedy_agent = QLearningAgent(k, alpha=1/n, epsilon=0.0)
greedy_rewards = train(greedy_agent, bandit, n)

plt.plot(np.cumsum(epsilon_greedy_rewards), label='Epsilon-Greedy (epsilon=0.1)')
plt.plot(np.cumsum(greedy_rewards), label='Greedy (epsilon=0.0)')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title(f'Comparison of Epsilon-Greedy and Greedy Strategies for {k}-Armed Bandit')
plt.legend()
plt.show()
