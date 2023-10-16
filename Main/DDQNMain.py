from Net.ddpg import DDPG
from Socket.controller import Controller

import keras.models


if __name__ == '__main__':
    # 参数
    w1 = 0.5
    w2 = 0.5
    th_under = 0.2

    # 模拟环境有关参数初始化
    # 观测空间
    state_space = (3, 4)
    action_space = 67

    # 实例化DDQN类
    # deepNetwork = DQNAgent(state_space, action_space)
    ddpgNetwork1 = DDPG(1, 3, 1)
    ddpgNetwork2 = DDPG(1, 3, 1)
    lstmNetwork1 = keras.models.load_model('../LSTM/model1.h5')
    lstmNetwork2 = keras.models.load_model('../LSTM/model2.h5')
    # DQN执行
    Controller("localhost", 32100, ddpgNetwork1, ddpgNetwork2, lstmNetwork1, lstmNetwork2)
    # receive_server("localhost", 32100, deepNetwork)
