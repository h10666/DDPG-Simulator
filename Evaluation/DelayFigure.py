import matplotlib.pyplot as plt
import numpy as np

with open('D:\HYZ\Simulation Platform\DynamicMobSim\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\w1=0.5-w2=0.5-th_under=0.2-evaluation.txt', 'r') as f:
    results = f.readlines()
    col = []
    for i in range(1, 721):
        # col.append(float(app1-results[i].split('\t')[0]))
        col.append(float(results[i].split('\t')[0]))
    m = np.max(col)
    n = np.min(col)
    plt.xlim(xmax=721, xmin=0)
    plt.ylim(ymax=190, ymin=155)
    plt.plot(range(1, 721), col)
    plt.show()