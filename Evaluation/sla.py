import matplotlib.pyplot as plt
import numpy as np

fognode_id = 5
with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\server'
          +str(fognode_id)+'_sla.txt', 'r') as f:
    results = f.readlines()

    steps = 0
    avg_time = []
    sla_rate = []
    while steps < 1439*2:
        temp = results[steps].split('\t')
        if int(temp[3]) == 0:
            sla_rate.append(1)
        else:
            sla_rate.append(int(temp[4]) / int(temp[3]))
        if len(temp) > 5:
            avg_time.append(float(temp[5]))
        else:
            avg_time.append(0)
        steps = steps + 2

    plt.plot(avg_time, '.', color='red', label='response_time')
    plt.plot(sla_rate, '.', color='blue', label='sla')
    # plt.plot([0.95] * len(sla_rate), color='black', label='base_line')
    plt.title(label='server_' + str(fognode_id))
    plt.xlabel(xlabel='episode')
    plt.ylabel(ylabel='ratio')
    plt.legend()
    # plt.savefig('./sla/server_' + str(i//2) + '_app_' + str(i % 2+1) + '_sla.png')
    plt.show()





