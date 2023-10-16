import os
import time
from collections import deque
import pickle
import threading
from Algorithm.ddpg_learner import DDPG
from Algorithm.models import Actor, Critic
from Algorithm.memory import Memory
from Algorithm.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from gym import spaces, logger
from baselines import logger
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class WebAgent:
    def __init__(self, seed=None, action_space=None, observation_space=None,
                 network=None,
                 num_timesteps=1e6,
                 noise_type='ou_0.1',  #'adaptive-param_0.2' ou_0.2
                 normalize_returns=False,
                 normalize_observations=False,
                 critic_l2_reg=1e-2,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 popart=False,
                 gamma=0.99,
                 clip_norm=None,
                 batch_size=64,  # per MPI worker
                 tau=0.0,
                 reward_scale=1.0,
                 runspace=None,
                 loadmodel=True,
                 buffer_size=2000,
                 **network_kwargs):
        self.buffer_size = buffer_size
        self.runspace = runspace
        self.loadmodel = loadmodel
        self.seed = seed
        self.observation_space = observation_space
        self.action_space = action_space
        self.network = network
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        # 模拟环境有关参数初始化
        #self.action_space = spaces.Discrete(31)

        # observation space
        # arrivalrate, queuing length
        high = [np.finfo(np.float32).max, np.finfo(np.float32).max]
        low = [0, 0]
        high = np.array(high)
        low = np.array(low)
        self.observation_space = spaces.Box(low, high)
        # action space capacity
        low = np.array([0])
        high = np.array([1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.lock = threading.Lock()
        self.NewSampleNumber = 0
        self.step = 0

        nb_actions = self.action_space.shape[-1]


        # assert (np.abs(self.action_space.low) == self.action_space.high).all()  # we assume symmetric actions 类似 [-1,1].

        self.memory = Memory(limit=int(self.buffer_size), action_shape=self.action_space.shape,
                        observation_shape=self.observation_space.shape)
        self.critic = Critic(network=self.network, **network_kwargs)
        self.actor = Actor(nb_actions, network=self.network, **network_kwargs)
        self.status = 0
        action_noise = None
        param_noise = None
        if noise_type is not None:
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                         desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                    k = 1
                    while k < 100:
                        print('testing noise', action_noise())
                        k = k + 1
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                                sigma=float(stddev) * np.ones(nb_actions))
                    k = 1
                    while k < 100:
                        print(action_noise())
                        k = k + 1
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        max_action = self.action_space.high
        logger.info('scaling actions by {} before executing in env'.format(max_action))

        self.agent = DDPG(self.actor, self.critic, self.memory, self.observation_space.shape, self.action_space.shape,
                     gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                     normalize_observations=normalize_observations,
                     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                     critic_l2_reg=critic_l2_reg,
                     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                     reward_scale=reward_scale, runspace=self.runspace)
        logger.info('Using agent with the following configuration:')
        logger.info(str(self.agent.__dict__.items()))

        eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        self.sess = U.get_session()
        self.sess.as_default()    # add czc
        self.sess.graph.as_default()
        # Prepare everything.
        self.agent.initialize(self.sess)

        if self.loadmodel:
            self.agent.loadmodel()

        self.sess.graph.finalize()

        self.agent.reset()

        self.FinishTraining = False


    def store_sample(self, observation, action, reward, newobservation, finishstatus):
        while self.NewSampleNumber >= self.batch_size:
            print("----------------------------waiting-----------------------------------")
            time.sleep(1)

        print('self.step=', self.step)
        print('self.NewSampleNumber=', self.NewSampleNumber)

        self.lock.acquire()
        self.NewSampleNumber = self.NewSampleNumber + 1
        self.lock.release()


        observation = np.array(observation).reshape([-1,self.observation_space.shape[-1]])
        action = np.array(action).reshape([-1, self.action_space.shape[-1]])
        reward = np.array(reward).reshape([-1, 1])
        newobservation = np.array(newobservation).reshape([-1, self.observation_space.shape[-1]])
        finishstatus = np.array(finishstatus).reshape([-1, 1])
        with self.sess.as_default():  #跨线程时需要明确指定使用的session
            with self.sess.graph.as_default():
                # the batched data will be unrolled in memory.py's append.
                self.agent.store_transition(observation, action, reward, newobservation, finishstatus)
        return True

    def getAction(self, observation):
        action, q, _, _ = self.agent.step(observation, apply_noise=True, compute_Q=True)

        return action
    def runthread(self):
        self.learn(total_timesteps=1e6)

    def learn(self,
              total_timesteps=None,
              nb_epochs=None,  # with default settings, perform 1M steps total
              nb_epoch_cycles=20,
              nb_rollout_steps=100,
              render=False,
              render_eval=False,
              nb_train_steps=5000,  # per epoch cycle and MPI worker,
              nb_eval_steps=100,
              eval_env=None,
              param_noise_adaption_interval=50,
              **network_kwargs):

        set_global_seeds(self.seed)

        if total_timesteps is not None:
            assert nb_epochs is None
            nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
        else:
            nb_epochs = 500



        start_time = time.time()


        totaltrained_num = 0
        with self.sess.as_default():
            with self.sess.graph.as_default():
                while not self.FinishTraining:
                    time.sleep(10)
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []
                    if self.NewSampleNumber >= self.batch_size:
                        self.lock.acquire()
                        traintimes = self.NewSampleNumber
                        totaltrained_num = totaltrained_num + traintimes
                        self.step = self.step + traintimes
                        # 增加人工样本
                        if self.step >= 40:
                            #self.memory.generateBoundarySamples()
                            # 更新噪声值
                            # if self.step == 100:
                            #     self.agent.action_noise = NormalActionNoise(mu=np.zeros(self.action_space.shape[-1]),
                            #                                                 sigma=float(0.2) * np.ones(
                            #                                                 self.action_space.shape[-1]))
                            # elif self.step == 140:
                            #     self.agent.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_space.shape[-1]),
                            #                                                 sigma=float(0.1) * np.ones(
                            #                                                 self.action_space.shape[-1]))
                            # if self.step == 200:
                            #     self.agent.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_space.shape[-1]),
                            #                                                 sigma=float(0.05) * np.ones(
                            #                                                 self.action_space.shape[-1]))
                            # elif self.step == 600:
                            #     self.agent.action_noise = OrnsteinUhlenbeckActionNoise(
                            #         mu=np.zeros(self.action_space.shape[-1]),
                            #         sigma=float(0.01) * np.ones(
                            #             self.action_space.shape[-1]))
                            self.agent.reset() #将行为网络复制给putubed 行为网络
                            self.status = 1
                            print("-----------------------------------start training------------------------------------")
                            for t_train in range(nb_train_steps):
                                if self.memory.nb_entries >= self.batch_size and t_train % param_noise_adaption_interval == 0:
                                    distance = self.agent.adapt_param_noise()
                                    epoch_adaptive_distances.append(distance)

                                cl, al = self.agent.train()
                                epoch_critic_losses.append(cl)
                                epoch_actor_losses.append(al)
                                self.agent.update_target_net()
                            print('epoch_actor_losses', epoch_actor_losses)
                            print('epoch_critic_losses', epoch_critic_losses)
                            print("-----------------------------------finish training------------------------------------")
                        self.status = 0
                        self.NewSampleNumber = 0
                        self.lock.release()
                        self.agent.savemodel()
                        print('-------------------------------------save model success---------------------------------')