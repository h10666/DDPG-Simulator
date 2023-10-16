import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DDPG-simulator\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\samples.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)
    rewards = []
    steps = 0
    reward = 0
    while steps < 360*18:
        temp = results[steps].split('\t')
        reward += float(temp[2][1:-1])
        steps = steps+1
        if steps % 50 == 0:
            rewards.append(reward/50)
            reward = 0

plt.plot(rewards, color='red')
plt.title(label='rewards')
plt.xlabel(xlabel='episode')
plt.ylabel(ylabel='rewards')
plt.legend()
# plt.savefig('./rewards.png')
plt.show()