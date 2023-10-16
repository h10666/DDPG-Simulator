import matplotlib.pyplot as plt

with open('D:\HYZ\Simulation Platform\DDPG-simulator2\modules\cloudsim\src\main\java\org\cloudbus\\fog\\resultfiles\evaluation\\samples.txt', 'r') as f:
    samples = f.readlines()
    actions = []
    rewards = []
    for sample in samples:
        info = sample.split('\t')
        arrive_rate = float(info[0].split(',')[0][1:])
        action = float(info[1][1:-1])
        reward = float(info[2][1:-1])
        if arrive_rate>100 and arrive_rate<=140:
            actions.append(action)
            rewards.append(reward)

    plt.plot(actions, rewards, 'ro', color='blue', label='reward')
    plt.title(label='analysis')
    plt.xlabel(xlabel='episode')
    plt.ylabel(ylabel='reward')
    plt.legend()
    # plt.savefig('./container_num/server_' + str(i//2) + '_app_' + str(i % 2+1) + '_container_num.png')
    plt.show()