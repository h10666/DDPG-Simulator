import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DDPG-simulator\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\samples.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)

    for i in range(0, 18):
        steps = i
        rewards = []
        while steps < 1000*18:
            temp = results[steps].split('\t')
            rewards.append(float(temp[2][1:-1]))
            steps = steps+18

        plt.plot(rewards, color='blue', label='reward')
        plt.title(label='server_' + str(i//2) + '_app_' + str(i % 2+1))
        plt.xlabel(xlabel='episode')
        plt.ylabel(ylabel='reward')
        plt.legend()
        # plt.savefig('./container_num/server_' + str(i//2) + '_app_' + str(i % 2+1) + '_container_num.png')
        plt.show()