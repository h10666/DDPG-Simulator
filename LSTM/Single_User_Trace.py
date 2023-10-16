class SingleUserTrace:
    def __init__(self, trace_name):
        self.la_splits = 3
        self.lo_splits = 3
        self.min_latitude = 37.76
        self.max_latitude = 37.80
        self.min_longitude = -122.44
        self.max_longitude = -122.40
        self.latitude_unit = (self.max_latitude - self.min_latitude) / self.la_splits
        self.longitude_unit = (self.max_longitude - self.min_longitude) / self.lo_splits
        # 用户轨迹文件路径
        self.trace_name = trace_name
        # 用户轨迹信息[纬度， 经度， 是否载客，Unix时间戳]
        self.trace_info = list()
    # 加载用户轨迹
    def load_trace_info(self):
        with open(self.trace_name, 'r') as f:
            traces = f.readlines()
            # print(int(traces[-1].split('\n')[0].split(' ')[3]))
            for trace in traces[::-1]:
                info = trace.split('\n')[0].split(' ')
                latitude = float(info[0])
                longitude = float(info[1])
                fog_node_id = int(((latitude-self.min_latitude)//self.latitude_unit)*self.lo_splits
                                  + (longitude-self.min_longitude)//self.longitude_unit)
                occupied = int(info[2])
                timestamps = int(info[3])-1211018404  # 从5.17 18：04开始
                self.trace_info.append([timestamps, fog_node_id, occupied])

    # 获取用户轨迹详情
    def get_trace_info(self):
        self.load_trace_info()
        return self.trace_info
