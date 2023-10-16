import socket
import threading
import json
from Evaluation.RewardRecord import RewardRecord
from Evaluation.LossRecord import LossRecord
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Controller:
    epsilon = 0.9
    epsilon_decay = 0.995  # 0.995
    final_epsilon = 0.01
    batch_size = 32
    current_step = 0
    update_steps = 40

    def __init__(self, rec_ip, rec_port, ddpg1, ddpg2, lstm1, lstm2):
        self.rec_ip = rec_ip
        self.rec_port1 = rec_port
        # threading.Thread(target=self.server_listen_client_obs, args=(rec_ip, rec_port,)).start()
        self.rev_obs_count = 0
        self.current_episode = 0
        self.accumulated_rewards = 0
        self.accumulated_losses = 0
        self.ddpg1 = ddpg1
        self.ddpg2 = ddpg2
        self.lstm1 = lstm1
        self.lstm2 = lstm2
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

    def sc_fit_transform(self, raw_data):
        # 将所有数据归一化为0-1的范围
        sc = MinMaxScaler(feature_range=(0, 1))
        dataset_transform = sc.fit_transform(X=raw_data)
        # 归一化后的数据
        return sc, np.array(dataset_transform)

    # 接收obs, 返回action
    def receive_obs(self, conn):
        jsonData = conn.recv(1024).decode('utf-8')
        # print(jsonData)
        flag = True
        while flag:
            try:
                data = json.loads(jsonData)
                print(data)
                send_data = {}
                data_type = data['type']
                receive_data = data['data']
                if data_type == 'actionRequest':
                    states = list(receive_data['state'])
                    app1_container_num = states[0]
                    app2_container_num = states[1]
                    app1_user_num = states[2]
                    app2_user_num = states[3]
                    app1_user_num_predict = self.lstm1.predict(app1_user_num)
                    app2_user_num_predict = self.lstm2.predict(app2_user_num)
                    node_num = len(states[0])
                    for i in range(node_num):
                        pass
                    state1 = np.array(states[0]).reshape((self.ddpg1.env_dim, ))
                    state2 = np.array(states[1]).reshape((self.ddpg2.env_dim, ))
                    action1 = self.ddpg1.policy_action(state1)
                    action2 = self.ddpg2.policy_action(state2)
                    send_data['type'] = 'action'
                    send_data['action'] = (float(action1[0]), float(action2[0]))
                    print('---------------------' + send_data['type'] + '------------------------')
                    # print('state: ', state)
                    # print('action: ', list(action[0]))
                    conn.send(json.dumps(send_data).encode('utf-8'))
                elif data_type == 'sample':
                    self.current_step += 1
                    # print(receive_data)
                    state = np.asarray(receive_data['state']).reshape((self.ddpg.env_dim, ))  # shape(3, 4)
                    # print('state:'+str(state))
                    # print(state)
                    # print(np.asarray(state))
                    action = np.asarray(receive_data['action'], dtype=int).reshape(())
                    reward = np.asarray(float(receive_data['reward'])).reshape(())
                    next_state = np.asarray(receive_data['next_state']).reshape((self.ddpg.env_dim, ))
                    # print('next_state:' + str(next_state))
                    done = np.asarray(int(receive_data['done'])).reshape(())
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
                    # 每一百步保存一次模型
                    # if self.current_episode % 100 == 0:
                    #     self.deepqNetwork.save('D:/HYZ/PyProjects/RLPlatform-v0.0/TrainedNetModel/trainedmodel-' +
                    #                            str(int(self.current_episode/100)))
                    # 保存样本
                    self.ddpg.memorize(state, action, reward, done, next_state)
                    # 回放(训练)
                    # print('memory.size(): ' + str(self.deepqNetwork.memory.size()))
                    if self.ddpg.buffer.size() > self.batch_size:
                        self.ddpg.replay(self.batch_size)
                    # 记录损失
                    # self.accumulated_losses += self.deepqNetwork.history.losses[0]
                    # # 更新目标网络
                    # if self.current_step % self.update_steps == 0:
                    #     self.Algorithm.update_target_model()
                    send_data['type'] = 'ack'
                    conn.send(json.dumps(send_data).encode('utf-8'))
                    # 数据完整收到，退出
                    # print('-----------------------sample received successfully------------------------')
                flag = False
            except Exception as e:
                print(e)
                send_data = {}
                send_data['type'] = 'resend'
                conn.send(json.dumps(send_data).encode('utf-8'))
                jsonData = conn.recv(1024).decode('utf-8')

        conn.close()


# if __name__ == '__main__':
#     comm1 = receive_server('192.168.2.190', 32088)