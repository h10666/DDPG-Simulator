import matplotlib.pyplot as plt
import numpy as np

fognode_id = 5
app_type = 1
with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\server'
          +str(fognode_id)+'_samples.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)

    for i in range(0, 1):
        steps = 1
        request_arrival_rate = []
        while steps < 1439*2:
            temp = results[steps].split('\t')
            request_arrival_rate.append(float(temp[0].split(',')[0][1:]))
            steps = steps+2

        plt.plot(request_arrival_rate, color='blue', label='arrival_rate')
        plt.title(label='server'+str(fognode_id)+'_app'+str(app_type))
        plt.xlabel(xlabel='time')
        plt.ylabel(ylabel='rate')
        plt.legend()
        # plt.savefig('./server8_app0_requests.png')
        plt.show()