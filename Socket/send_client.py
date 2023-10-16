import socket
import struct
import pickle
import json
import numpy as np
class send_client:

    def __init__(self, send_ip, send_port1, send_port2, send_port3):
        self.send_ip = send_ip
        self.send_port1 = send_port1
        self.send_port2 = send_port2
        self.send_port3 = send_port3
        self.obs_client = self.create_client('obs_client', send_ip, send_port1)
        self.sample_client = self.create_client('sample_client', send_ip, send_port2)
        # 与环境进行通信
        # self.env_client = self.create_client('env_client', send_ip, send_port3)
        # print('hhhh')

    def create_client(self, client_name, ip, port):
        print('创建客户端(' + client_name + ')...')
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            # 成功返回0
            if (client.connect_ex((ip, port)) == 0):
                print(client_name + "创建成功")
                break
        client.settimeout(10000)
        return client


    # 发送obs, 返回action
    def send_obs(self, obs, update_eps):
        obs_update = []
        obs_update.append(obs)
        obs_update.append(update_eps)
        obs_update_str = pickle.dumps(obs_update)

        try:
            self.obs_client.send(struct.pack('i', len(obs_update_str)))  # 固定发送4个字节的报头，告知报文的长度
            self.obs_client.send(obs_update_str)

            header = self.obs_client.recv(4)
            length, = struct.unpack('i', header)
            back_data = self.obs_client.recv(length)
            action = pickle.loads(back_data)
            return action
        except ConnectionResetError:
            self.obs_client.close()
            self.obs_client = self.create_client(self.send_ip, self.send_port1)
            self.obs_client.send(struct.pack('i', len(obs_update_str)))  # 固定发送4个字节的报头，告知报文的长度
            self.obs_client.send(obs_update_str)

            header = self.obs_client.recv(4)
            length, = struct.unpack('i', header)
            back_data = self.obs_client.recv(length)
            action = pickle.loads(back_data)
            return action

    # 发送sample[obs, action, rew, new_obs, done]
    def send_sample(self, obs, action, rew, new_obs, done):
        sample = []
        sample.append(obs)
        sample.append(action)
        sample.append(rew)
        sample.append(new_obs)
        sample.append(done)

        sample_str = pickle.dumps(sample)
        try:
            self.sample_client.send(struct.pack('i', len(sample_str)))  # 固定发送4个字节的报头，告知报文的长度
            self.sample_client.send(sample_str)
        except ConnectionResetError:
            self.sample_client.close()
            self.sample_client = self.create_client(self.send_ip, self.send_port2)
            self.sample_client.send(struct.pack('i', len(sample_str)))  # 固定发送4个字节的报头，告知报文的长度
            self.sample_client.send(sample_str)

        try:
            header = self.sample_client.recv(4)
            length, = struct.unpack('i', header)
            back_data = self.sample_client.recv(length)
            print(back_data)
        except ConnectionResetError:
            self.sample_client.close()
            self.sample_client = self.create_client(self.send_ip, self.send_port2)
            # header = self.sample_client.recv(4)
            # length, = struct.unpack('i', header)
            # back_data = self.sample_client.recv(length)
            # print(back_data)

    # 环境相关操作(发送操作指令, 接收返回值)
    def send_env(self, data):
        self.env_client = self.create_client('env_client', self.send_ip, self.send_port3)
        data = (data+'\n').encode('utf-8')
        try:
            self.env_client.send(data)
            ack = self.env_client.recv(1024*1024).decode('utf-8')
            return ack
        except ConnectionResetError:
            self.env_client.close()
            self.env_client = self.create_client(self.send_ip, self.send_port3)
            self.env_client.send(data)
            ack = self.env_client.recv(1024*1024).decode('utf-8')
            return ack
