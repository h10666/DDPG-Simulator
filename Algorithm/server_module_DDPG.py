import struct
import socket
import threading
import numpy as np
import pickle
import json
import time
from multiprocessing import Queue
from baselines.common.schedules import LinearSchedule


# coding=utf-8
class server_module:

    def __init__(self, rec_ip, rec_port1, rec_port2, deep_web):
        self.rec_ip = rec_ip
        self.rec_port1 = rec_port1
        self.rec_port2 = rec_port2
        self.deep_web = deep_web
        self.sample_num = 0
        threading.Thread(target=self.server_listen_client_obs, args=(rec_ip, rec_port1,)).start()

    # 监听指定端口，接收obs
    def server_listen_client_obs(self, ip, port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            204800
        )
        server.bind((ip, port))
        server.listen(60)
        print("开始监听客户端连接：")
        while True:
            conn, addr = server.accept()
            print(conn)
            try:
                data = conn.recv(1024)
                print('Receivedata:', data, len(data))
                if len(data) == 0:
                    # time.sleep(1000)
                    continue

                datadecoded = data.decode(encoding='UTF-8')  # 'UTF-8'
                print('datadecoded:', datadecoded)

                obs_update = json.loads(datadecoded, strict=False)
                #algorithmtype = obs_update["algorithmtype"]

                print('Algorithm: DDPG')
                print(obs_update, type(obs_update), obs_update["messagetype"])
                if (self.deep_web.status == 1 and obs_update["messagetype"] == "samples"):
                    reply = {'message': 'updating'}
                    action_str = json.dumps(reply, ensure_ascii=False)
                    print("action_str", action_str)
                    action_str_encoded = action_str.encode(encoding='UTF-8', errors='strict')
                    print("action_str_encoded", action_str_encoded)
                    conn.send(action_str_encoded)

                    continue

                if (obs_update["messagetype"] == "getaction"):
                    # states = obs_update['data']
                    # actions1 = [0]*9
                    # actions2 = [0]*9
                    # for i in range(9):
                    #     obs1 = np.matrix([states[2][i], states[0][i]])
                    #     action1 = self.deep_web.getAction(obs1)
                    #     obs2 = np.matrix([states[3][i], states[1][i]])
                    #     action2 = self.deep_web.getAction(obs2)
                    #     actions1[i] = float(action1[0])
                    #     actions2[i] = float(action2[0])

                    obs = [obs_update["arrivalrate"], obs_update["queuelength"]]
                    obs = np.matrix(obs)

                    print("obs", obs, type(obs))

                    action = self.deep_web.getAction(obs)
                    print('action:', action, type(action))

                    # print('actions1:', actions1, type(actions1))
                    # print('actions2:', actions2, type(actions2))

                    # pickle

                    reply = {'message': 'action', 'action': float(action[0])}
                    # reply = {'message': 'action', 'action1': str(actions1), 'action2': str(actions2)}
                    action_str = json.dumps(reply, ensure_ascii=False)
                    print("action_str", action_str)
                    action_str_encoded = action_str.encode(encoding='UTF-8', errors='strict')
                    print("action_str_encoded", action_str_encoded)
                    conn.send(action_str_encoded)
                elif (obs_update["messagetype"] == "samples"):
                    # [obs, action, rew, new_obs, done]
                    obs = [obs_update["arrivalrate"], obs_update["lastqueuinglength"]]
                    action = obs_update["action"]
                    rew = obs_update["reward"]
                    new_obs = [obs_update["newarrivalrate"],obs_update["currentqueuinglength"]]
                    done = obs_update["done"]



                    floatobs = [float(i) for i in obs]

                    floatnew_obs = [float(i) for i in new_obs]

                    if self.deep_web.store_sample(floatobs, action, float(rew), floatnew_obs, int(done)):
                        reply = {'message': 'confirm', 'status': 'ok'}
                    else:
                        reply = {'message': 'updating'}

                    action_str = json.dumps(reply, ensure_ascii=False)
                    print("action_str", action_str)
                    action_str_encoded = action_str.encode(encoding='UTF-8', errors='strict')
                    print("action_str_encoded", action_str_encoded)
                    conn.send(action_str_encoded)
                    self.sample_num += 1
                    print('sample_num=', self.sample_num)
                else:
                    print("unrecognized messagetype:", obs_update["messagetype"])
                # conn.send(struct.pack('i', len(action_str)))
                # conn.send(action_str)

            except UnicodeDecodeError:
                print('receive_obs UnicodeDecodeError')
                pass

            # print(data)
