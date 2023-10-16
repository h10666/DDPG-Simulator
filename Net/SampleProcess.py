import numpy as np

class SampleProcess:
    def __init__(self, deepqNetwork=None, env=None):
        self.deepqNetwork = deepqNetwork
        self.env = env
        self.epsilon = 0.9
        self.batch_size = 10
        self.update_steps = 100

    # store sample in the replay buffer
    def store_sample(self, state, action, reward, new_state, done):
        self.deepqNetwork.memorize(state, action, reward, new_state, done)
        return True
    def samplethread(self):
        self.sampling(episodes=1000, sample_steps=200)
    def sampling(self, episodes=1000, sample_steps=200):
        action_record_list = []
        reward_record_list = []
        interval = 10

        for episode in range(episodes):
            action_list = []
            reward_list = []
            print("--------------------------Begin to reset-------------------------")
            state = self.env.reset()
            print('/////////////////////episode:' + str(episode) + '////////////////////////')
            for t in range(sample_steps):
                print('sample_steps:', t)
                Qm = self.deepqNetwork.getQm()
                if(np.random.rand()>self.epsilon and str(state) in Qm.keys()):
                    action = Qm[str(state)][0]
                else:
                    # 启发式算法获得动作
                    action = self.env.getActions()
                print('action', action)
                new_state, rew, done, _ = self.env.step(action)

                action_list.append(action)
                reward_list.append(rew)
                self.store_sample(state, action, rew, new_state, done)
                state = new_state

                if done:
                    #输出到文件
                    # doc = open('D:/HYZ/Runspace/output23/validate_out4.txt', 'a')
                    # print("/////////////////////////////////////////////", file=doc)
                    # print(episode_rewards, file=doc)
                    # doc.close()

                    #记录动作
                    doc = open('D:/HYZ/Runspace/output23/validate_action_out4.txt', 'a')
                    print("/////////////////////////////////////////////", file=doc)
                    print(action_record_list, file=doc)
                    doc.close()

                    # 记录reward
                    doc = open('D:/HYZ/Runspace/output16/reward_record2_4.txt', 'a')
                    print("/////////////////////////////////////////////", file=doc)
                    print(reward_record_list, file=doc)
                    doc.close()


                    break
                if t % interval == 0:
                    print("/////////////////////////////////////////////t=", t)
                    print("current_state:", state)
                    print("current_action:", action)
                    print("reward:", rew)
                # 更新目标网络
                if t % self.update_steps == 0:
                    self.deepqNetwork.update_target_model()
                # 经验回放（训练网络）
                if len(self.deepqNetwork.memory) > self.batch_size:
                    self.deepqNetwork.replay(self.batch_size)

        return True