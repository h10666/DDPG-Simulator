import numpy as np
import os
from LSTM.Single_User_Trace import SingleUserTrace
import time

class DataPreparing:
    def __init__(self, user_num):
        self.trace_dir = 'D:\\HYZ\\PyProjects\\RLPlatform-v2.0-user-distribution\\LSTM\\cabtrace\\'
        self.fog_node_num = 9
        self.simulation_time = 20*24*60*60
        self.user_num = user_num
        self.user_trace_details = []
        self.app_train_x1 = []
        self.app_train_x2 = []

    # 加载用户轨迹信息
    def load_user_trace(self):
        trace_files = os.listdir(self.trace_dir)
        for i in range(self.user_num):
            single_user_trace = SingleUserTrace(self.trace_dir+trace_files[i])
            self.user_trace_details.append(single_user_trace.get_trace_info())

    # 获取用户开始移动和停止时间
    def get_trace_time(self):
        for user_trace in self.user_trace_details:
            print([user_trace[0][0], user_trace[-1][0]])

    # 得到训练数据以及标签
    def get_data(self):
        self.load_user_trace()
        # 用户轨迹索引
        index = [0]*self.user_num
        for t in range(0, self.simulation_time, 60):
            # 应用一的样本及标签
            each_time_slot_info_app1_x = [0]*self.fog_node_num
            # 应用二的样本及标签
            each_time_slot_info_app2_x = [0]*self.fog_node_num
            for i in range(self.user_num):
                user_trace = self.user_trace_details[i]
                # 用户在t时刻的轨迹信息
                cur_index = index[i]
                # 保证用户拥有下一时刻的轨迹
                if cur_index < len(user_trace):
                    # 用户cur_index处的轨迹在t slot内
                    if user_trace[cur_index][0] < t+60:
                        node_id_t = user_trace[cur_index][1]
                        request_type_t = user_trace[cur_index][2]
                        # 在地图的范围内
                        if 0 <= node_id_t < self.fog_node_num:
                            # 对应边缘节点用户数加一
                            if request_type_t == 0:
                                each_time_slot_info_app1_x[node_id_t] += 1
                            else:
                                each_time_slot_info_app2_x[node_id_t] += 1
                        index[i] += 1
            self.app_train_x1.append(each_time_slot_info_app1_x)
            self.app_train_x2.append(each_time_slot_info_app2_x)
        return self.app_train_x1, self.app_train_x2

