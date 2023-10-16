import matplotlib.pyplot as plt

# x与y轴数据
x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# y = [-0.12379783, -0.32346034, -0.515606, -0.6719332, -0.82425094, -0.9205989, -0.9754142, -1.0302296, -1.083731, -1.1366149, -1.1894996]

y = []
file_object = open("compute_Q.txt" ,'r')
try:
    lines = file_object.readlines()
    for line in lines:
        if line.__contains__("compute_Q"):
            print(line)
            print(line[13:-3])
            y.append(float(line[13:-3]))
finally:
    file_object.close()

print(y)
# 绘图
plt.plot(x,  # x轴数据
         y,  # y轴数据
         linestyle='-',  # 折线类型,'.''-.'
         linewidth=3,  # 折线宽度
         color='b',  # 折线颜色
         markersize=5,  # 点的大小
         markerfacecolor='g')  # 点的填充色

# 添加标题和坐标轴标签
plt.xlabel('x-action')
plt.ylabel('y-Q')
plt.show()
