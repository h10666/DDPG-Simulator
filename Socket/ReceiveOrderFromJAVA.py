import socket
import threading
import json
from Evaluation.RewardRecord import RewardRecord
from Evaluation.LossRecord import LossRecord
from Env.Constants import ActionMap
import random
import numpy as np

class receive_server:
    epsilon = 0.9
    epsilon_decay = 0.995  # 0.995
    final_epsilon = 0.01
    batch_size = 32
    current_step = 0
    update_steps = 40
    action_map = ActionMap()

    def __init__(self, rec_ip, rec_port, deepqNetwork):
        self.rec_ip = rec_ip
        self.rec_port1 = rec_port
        # threading.Thread(target=self.server_listen_client_obs, args=(rec_ip, rec_port,)).start()
        self.rev_obs_count = 0
        self.current_episode = 0
        self.accumulated_rewards = 0
        self.accumulated_losses = 0
        self.deepqNetwork = deepqNetwork
        self.accumulated_rewards_records = RewardRecord('./rewards/accumulated_rewards.txt')
        self.accumulated_losses_records = LossRecord('./losses/loss.txt')
        self.server = self.server_listen_client_obs(rec_ip, rec_port)
        # self.env = env

    # 监听指定端口，接收obs
    def server_listen_client_obs(self, ip, port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            1024*1024
        )
        server.bind((ip, port))
        server.listen(60)
        print("obs端开始监听客户端连接：")
        while True:
            conn, addr = server.accept()
            # print("obs端连接成功")
            # threading.Thread(target=self.receive_obs, args=(conn,)).start()
            self.receive_obs(conn)

    # 产生随机动作
    def get_random_action(self, states):
        random_action = []
        for i in range(states.shape[0]):
            random_action.append(random.randint(1, 67))
        # inc = 10
        # for i in range(states.shape[0]):
        #     if states[i][0] < inc:
        #         random_action.append(random.randint(0, states[i][0] + inc))
        #     elif states[i][0] > 67-inc:
        #         random_action.append(random.randint(states[i][0] - inc, 67))
        #     else:
        #         random_action.append(random.randint(states[i][0] - inc, states[i][0] + inc))

        return np.array(random_action)

    # 接收obs, 返回action
    def receive_obs(self, conn):
        jsonData = conn.recv(1024).decode('utf-8')
        # print(jsonData)
        flag = True
        while flag:
            try:
                data = json.loads(jsonData)
                # print(data)
                sendData = {}
                data_type = data['type']
                receiveData = data['data']
                if data_type == 'actionRequest':
                    state = np.asarray(receiveData['state']).reshape(self.deepqNetwork.input_shape)
                    # print(state)
                    if random.random() < self.epsilon:
                        action = self.get_random_action(state).reshape(1, state.shape[0])
                        sendData['type'] = 'RandomActions'
                    else:
                        action = np.argmax(self.deepqNetwork.model.predict(state.reshape((1,)+state.shape)), axis=-1)\
                            .reshape(1, state.shape[0]) + 1  # 网络给出的动作范围为[0,67)，所以+1
                        sendData['type'] = 'NetworkActions'
                    sendData['actions'] = action[0].tolist()
                    print('---------------------' + sendData['type'] + '------------------------')
                    # print('state: ', state)
                    # print('action: ', list(action[0]))
                    conn.send(json.dumps(sendData).encode('utf-8'))
                elif data_type == 'sample':
                    self.current_step += 1
                    # print(receiveData)
                    state = np.asarray(receiveData['state']).reshape(self.deepqNetwork.input_shape)  # shape(3, 4)
                    # print('state:'+str(state))
                    # print(state)
                    # print(np.asarray(state))
                    action = np.asarray(receiveData['action'], dtype=int).reshape((self.deepqNetwork.input_shape[0],))
                    reward = np.asarray(float(receiveData['reward'])).reshape(())
                    nextState = np.asarray(receiveData['nextState']).reshape(self.deepqNetwork.input_shape)
                    # print('next_state:' + str(nextState))
                    done = np.asarray(int(receiveData['done'])).reshape(())
                    # print('action: ', action)
                    print('reward: ', reward)
                    # 记录回报
                    # self.recorder.record(reward)
                    self.accumulated_rewards += reward
                    self.rev_obs_count += 1
                    # if self.rev_obs_count % 720 == 0:
                    if done == 1:
                        # self.accumulated_rewards_records.record(self.accumulated_rewards/self.rev_obs_count*720)
                        self.accumulated_rewards_records.record(self.accumulated_rewards)
                        self.accumulated_losses_records.record(self.accumulated_losses)
                        self.accumulated_rewards = 0
                        self.accumulated_losses = 0
                        self.rev_obs_count = 0
                        self.current_episode += 1
                        if self.epsilon > self.final_epsilon:
                            self.epsilon *= self.epsilon_decay
                        # else:
                        #     self.epsilon = 10**-3
                    # 每一百步保存一次模型
                    if self.current_episode % 100 == 0:
                        self.deepqNetwork.save('D:/HYZ/PyProjects/RLPlatform-v0.0/TrainedNetModel/trainedmodel-' +
                                               str(int(self.current_episode/100)))
                    # 保存样本
                    self.deepqNetwork.memorize(state, action, reward, nextState, done)
                    # 回放(训练)
                    # print('memory.size(): ' + str(self.deepqNetwork.memory.size()))
                    if self.deepqNetwork.memory.size() > self.batch_size:
                        self.deepqNetwork.replay(self.batch_size)
                    else:
                        self.deepqNetwork.replay(self.deepqNetwork.memory.size())
                    # 记录损失
                    self.accumulated_losses += self.deepqNetwork.history.losses[0]
                    # # 更新目标网络
                    if self.current_step % self.update_steps == 0:
                        self.deepqNetwork.update_target_model()
                    sendData['type'] = 'ack'
                    conn.send(json.dumps(sendData).encode('utf-8'))
                    # 数据完整收到，退出
                    # print('-----------------------sample received successfully------------------------')
                flag = False
            except Exception as e:
                print(e)
                sendData = {}
                sendData['type'] = 'resend'
                conn.send(json.dumps(sendData).encode('utf-8'))
                jsonData = conn.recv(1024).decode('utf-8')

        conn.close()


# if __name__ == '__main__':
#     comm1 = receive_server('192.168.2.190', 32088)