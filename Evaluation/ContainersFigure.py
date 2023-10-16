import matplotlib.pyplot as plt
import numpy as np

fognode_id = 5
app_type = 1
max_fognode_num = 20
with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\server'
          +str(fognode_id)+'_samples.txt', 'r') as f:
    results = f.readlines()
    total_lines = len(results)

    for i in range(0, 1):
        steps = 1439*2+0
        container_num = []
        while steps < 1439*4:
            temp = results[steps].split('\t')
            container_num.append(max(0, round(float(temp[2][1:-1])*max_fognode_num)))
            steps = steps+2

        plt.plot(container_num, color='blue', label='container_num')
        plt.title(label='server'+str(fognode_id)+'_app'+str(app_type))
        plt.xlabel(xlabel='time')
        plt.ylabel(ylabel='num')
        plt.legend()
        # plt.savefig('./server8_app0_container_num.png')
        plt.show()