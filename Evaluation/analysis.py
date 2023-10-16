import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DDPG-simulator\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\requests_evaluation.txt', 'r') as f:
    results = f.readlines()
    response_time = 0
    power_consumption = 0
    scale_cost = 0
    steps = 0
    col = []
    for line in results:
        if line.startswith('#'):
            if steps != 0:
                col.append(response_time/steps)
                steps = 0
                response_time = 0
                power_consumption = 0
                scale_cost = 0
        else:
            temp = line.split('\t')
            response_time += float(temp[-3])
            power_consumption += float(temp[-2])
            scale_cost += float(temp[-1])
            steps += 1
with open('D:\HYZ\Simulation Platform\DynamicMobSim\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\temp.txt', 'r') as f:
        results = f.readlines()
        response_time = 0
        power_consumption = 0
        scale_cost = 0
        steps = 0
        col1 = []
        for line in results:
            if line.startswith('#'):
                if steps != 0:
                    col1.append(response_time / steps)
                    steps = 0
                    response_time = 0
                    power_consumption = 0
                    scale_cost = 0
            else:
                temp = line.split('\t')
                response_time += float(temp[-3])
                power_consumption += float(temp[-2])
                scale_cost += float(temp[-1])
                steps += 1

m = np.max(col)
n = np.min(col)
plt.xlim(xmax=len(col1)-1, xmin=0)
plt.ylim(ymax=1, ymin=0)
plt.xlabel('episodes')
plt.ylabel('response time')
plt.plot(range(0, len(col1)), col[0: len(col1)])
plt.plot(range(0, len(col1)), col1)
plt.show()