import socket
import json

class send_env:

    def __init__(self, send_ip, send_port):
        self.send_ip = send_ip
        self.send_port = send_port
        self.env_client = self.create_client('env_client', send_ip, send_port)

    def create_client(self, client_name, ip, port):
        # print('创建客户端(' + client_name + ')...')
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            # 成功返回0
            if (client.connect_ex((ip, port)) == 0):
                # print(client_name + "创建成功")
                break
        client.settimeout(10000)
        return client


    # 环境相关操作(发送操作指令, 接收返回值)
    def send_order(self, data):
        data = (data+'\n').encode('utf-8')
        try:
            self.env_client.send(data)
            ack = self.env_client.recv(1024*1024).decode('utf-8')
            self.env_client.close()
            return ack
        except ConnectionResetError:
            self.env_client.close()
            self.env_client = self.create_client(self.send_ip, self.send_port)
            self.env_client.send(data)
            ack = self.env_client.recv(1024*1024).decode('utf-8')
            self.env_client.close()
            return ack

if __name__ == '__main__':
    s = send_env('192.168.2.190', 32090)
    ack = s.send_order(json.dumps({'operation': 'reset'}))
    s = send_env('192.168.2.190', 32090)
    data_back = s.send_order(json.dumps({'operation': 'getAction'}))
    s = send_env('192.168.2.190', 32090)
    data = s.send_order(json.dumps({'operation': 'step', 'params': {'actions': json.loads(data_back)}}))
    data = json.loads(data)
    state = data_back[0:-1]
    reward = data_back[-1]
    print(data)
