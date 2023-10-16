from threading import *
from baselines.common import models
from Algorithm.WebAgent import WebAgent
from Algorithm.server_module_DDPG import server_module
from gym import spaces, logger
import tensorflow as tf

if __name__ == '__main__':
    #f = open("D:\Runspace\logs\python_" + str(0.1) + "Algorithm.log", 'a')
    #sys.stdout = f
    #sys.stderr = f

    #'ou_0.1' 'ou_0.05' 'ou_0.2'
    #实例化DeepQforWeb类 activation tf.nn.relu tf.tanh
    #num_hidden =50 100 200

    agent = WebAgent(network=models.mlp(num_layers=3, num_hidden=100, activation=tf.nn.relu),
                     num_timesteps=1e6, seed=2, batch_size=40, actor_lr=1e-4, critic_lr=1e-4, noise_type='ou_0.1',
                     runspace='D:/HYZ/PyProjects/DDPG-Simulator/Algorithm/model/ddpg', loadmodel=False,
                     buffer_size=2000)
    #神经网络开始学习
    Thread(target=agent.runthread).start()

   #主客户端，接收obs和update_eps参数，给出神经网络的指导动作
    server_module = server_module('127.0.0.1', 9000, 8081, agent)