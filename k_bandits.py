
import random as r
bandits = list(range(100000))
rewards = []
q = []
for i in range(len(bandits)):
    rewards.append(r.uniform(0, 1))
    q.append(rewards[i]+r.normalvariate(0,10))
#print(rewards)
#print(q)

#greedy method:
greedy_reward = 0
n = len(bandits)
reward_at_n_greedy = []
for i in range(n):
    #choose the bandit with the highest q-value 
    chosen_bandit = q.index(max(q))
    greedy_reward+=rewards[chosen_bandit]
    reward_at_n_greedy.append(greedy_reward)
    

#epsilon greedy method:
epsilon = 0.01
epsilon_greedy_reward = 0
expected_best_bandit = q.index(max(q))
reward_at_n = []
visited_bandits = []
expected_largest_reward_at_n = []
print()
m = 0
for i in range(n):
    expected_largest_reward_at_n.append(rewards[expected_best_bandit])
    #choose a random bandit with probability epsilon
    if m<len(bandits):
        if r.random() < epsilon:
            chosen_bandit = m
            epsilon_greedy_reward += rewards[chosen_bandit]
            reward_at_n.append(epsilon_greedy_reward)

            if rewards[chosen_bandit] > rewards[expected_best_bandit]:
                expected_best_bandit = chosen_bandit
        else:
            #choose the bandit with the highest q-value
            epsilon_greedy_reward += rewards[expected_best_bandit]
            reward_at_n.append(epsilon_greedy_reward)
        m+=1
    else:
        epsilon_greedy_reward+=rewards[expected_best_bandit]
        reward_at_n.append(epsilon_greedy_reward)

import matplotlib.pyplot as plt
import numpy as np
t = range(n)



plt.plot(reward_at_n,t,label='reward over time for epsilon greedy approach',color='blue')
plt.plot(reward_at_n_greedy,t,label='reward over time for greedy approach',color='red')
plt.show()
plt.plot(t,expected_largest_reward_at_n,label='expected largest reward at nth bandit pick',color='blue')

plt.show()

print(greedy_reward)
print(epsilon_greedy_reward)