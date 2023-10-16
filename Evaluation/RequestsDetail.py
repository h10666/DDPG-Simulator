import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\requests.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)
    step = 1
    request_arrival = []
    interval_num = 0
    times = 1*60
    for i in range(times):
        temp = results[i].split('\t')
        interval_num += int(temp[2])
        if i % step == 0:
            request_arrival.append(interval_num/step)
            interval_num = 0

    plt.plot(request_arrival, color='blue', label='arrival_rate')
    plt.title(label='server_5_app_1')
    plt.xlabel(xlabel='episode')
    plt.ylabel(ylabel='rate')
    plt.legend()
    # plt.savefig('./container_num/server_' + str(i//2) + '_app_' + str(i % 2+1) + '_container_num.png')
    plt.show()