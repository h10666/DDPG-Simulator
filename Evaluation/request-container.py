import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\samples.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)

    for i in range(0, 1):
        steps = i
        request_arrival_rate = []
        container_num = []
        while steps < 2500:
            temp = results[steps].split('\t')
            request_arrival_rate.append(float(temp[0].split(',')[0][1:]))
            container_num.append(max(0, round(float(temp[1][1:-1])*13)))
            steps = steps+1

        plt.plot(request_arrival_rate, color='red', label='arrival_rate')
        plt.plot(container_num, color='blue', label='container_num')
        plt.title(label='server_' + str(i//2) + '_app_' + str(i % 2+1))
        plt.xlabel(xlabel='episode')
        plt.ylabel(ylabel='nums/rate')
        plt.legend()
        # plt.savefig('./container_num/server_' + str(i//2) + '_app_' + str(i % 2+1) + '_container_num.png')
        plt.show()




