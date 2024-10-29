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
        reward = np.random.normal(self.q_true[action], 2)
        return reward

# Q-learning agent
class QLearningAgent:
    def __init__(self, k=10, alpha=0.1, epsilon=0.1, mode=None, c=1):
        self.k = k
        self.alpha = alpha  # Learning rate/step size
        self.epsilon = epsilon  # Exploration rate (0 for greedy method)
        self.q_estimates = np.zeros(k)  # Estimated action values
        self.action_counts = np.zeros(k)  # Counts for each action
        self.mode = mode  # Mode for agent's strategy
        self.avg_reward = 0 
        self.c = c  # Confidence level for UCB
        self.H = np.zeros(k)
    def softmax(self):
        """Compute the softmax probabilities for each action based on preferences."""
        return np.exp(self.H)/ np.sum(np.exp(self.H)) 
    def choose_action(self):
        if self.mode == 'pure_greedy':
            return np.argmax(self.q_estimates)
        
        elif self.mode == 'UCB':  # Upper Confidence Bound action selection
            # Calculate UCB values only for actions that have been selected at least once
            total_counts = np.sum(self.action_counts) + 1  # Total count of all actions
            ucb_values = self.q_estimates + self.c * np.sqrt(np.log(total_counts) / (self.action_counts + 1e-5))  # Avoid division by zero
            return np.argmax(ucb_values)
        elif self.mode == 'gradient':
            probabilities = self.softmax()
            return np.random.choice(self.k, p=probabilities)

        else:  # Epsilon-greedy
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.k)  # Explore
            else:
                return np.argmax(self.q_estimates)  # Exploit

    def update_q(self, action, reward):
        if self.mode == 'gradient':
            probabilities = self.softmax()

            self.avg_reward += (reward - self.avg_reward) / (np.sum(self.action_counts) + 1)

            # Update preferences based on reward
            for a in range(self.k):
                if a == action:
                    self.H[a] += self.alpha * (reward - self.avg_reward) * (1 - probabilities[a])
                else:
                    self.H[a] -= self.alpha * (reward - self.avg_reward) * probabilities[a]
        
        else:  # Regular Q-learning update
            self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])

        # Update action count for selected action
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

k = 10
n = 1000

bandit = KArmedBandit(k)

# Epsilon-greedy (e.g., epsilon = 0.1)
epsilon_greedy_agent = QLearningAgent(k, alpha=1/n, epsilon=0.1)
epsilon_greedy_rewards = train(epsilon_greedy_agent, bandit, n)

# Purely greedy (epsilon = 0)
greedy_agent = QLearningAgent(k, alpha=1/n, epsilon=0.0, mode='pure_greedy')
greedy_rewards = train(greedy_agent, bandit, n)

# UCB 
ucb_agent = QLearningAgent(k, alpha=1/n, c=1, mode='UCB')
ucb_rewards = train(ucb_agent, bandit, n)

agent = QLearningAgent(k, alpha=1/n, mode='gradient')
gradient_rewards = train(agent, bandit, n)


# Plot cumulative rewards for all strategies
plt.plot(np.cumsum(ucb_rewards), label='UCB')
plt.plot(np.cumsum(gradient_rewards), label='gradient method')
plt.plot(np.cumsum(epsilon_greedy_rewards), label='Epsilon-Greedy (epsilon=0.1)')
plt.plot(np.cumsum(greedy_rewards), label='Greedy (epsilon=0.0)')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title(f'Comparison of Epsilon-Greedy, Greedy, and UCB Strategies for {k}-Armed Bandit')
plt.legend()
plt.show()
