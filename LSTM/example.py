from pandas import read_csv
from matplotlib import pyplot
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# load dataset
listDataset = read_csv('jena_climate_2009_2016.csv')
listDataset = listDataset.drop(['Date Time', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'], axis=1)

columns = listDataset.columns



# 绘制初始图像，除去 'Date Time', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)' 数据
# values = listDataset.values
# groups = range(11)
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(listDataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()

def sc_fit_transform(nDlist):
    # 将所有数据归一化为0-1的范围
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset_transform = sc.fit_transform(X=nDlist)
    # 归一化后的数据
    return sc, np.asarray(dataset_transform)


# ###############################################################################
# 需要之前144次数据来预测下一次的数据，
timestep = 144
# 训练数据的大小
training_num = 14400
# 迭代训练10次
epoch = 10
# 每次取数据数量
batch_size = 200
# ###############################################################################
listDataset = np.asarray(listDataset)

# 生成训练数据集
xTrainDataset = listDataset[0:training_num]
print(xTrainDataset.shape)

yTrainDataset = listDataset[1:training_num + 1]
print(yTrainDataset.shape)

# 原始数据归一化
scTrainDataseX, xTrainDataset = sc_fit_transform(xTrainDataset)

scTrainDataseY, yTrainDataset = sc_fit_transform(yTrainDataset)
###############################################################################
# 生成lstm模型需要的训练集数据和
xTrain = []
for i in range(timestep, training_num):
    xTrain.append(xTrainDataset[i - timestep: i])
xTrain = np.asarray(xTrain)
print(xTrain.shape)

yTrain = []
for i in range(timestep, training_num):
    yTrain.append(yTrainDataset[i])
yTrain = np.asarray(yTrain)
print(yTrain.shape)


###############################################################################
# 构建网络，使用的是序贯模型
def _create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(144, 11)))
    model.add(Dense(11))
    # 进行配置
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


model = _create_model()
model.fit(x=xTrain, y=yTrain, epochs=epoch, batch_size=batch_size)
# index_start = 0
# for i in range(1000):
#     X_batch = xTrain[index_start:index_start + batch_size, :, :]
#     Y_batch = yTrain[index_start:index_start + batch_size, :]
#     index_start += batch_size
#     model.fit(X_batch, Y_batch)
#     if index_start >= xTrain.shape[0]:
#         index_start = 0
model.save('jena_model.h5')
###############################################################################
# 进行测试数据的处理
xTestDataset = listDataset[training_num: 15840 - 2]
scTesDatasetX, xTestDataset = sc_fit_transform(xTestDataset)
# 每次的下次开盘价是训练结果
yTestDataset = listDataset[training_num + 1: 15840 - 1]
scTestDataseY, yTestDataset = sc_fit_transform(yTestDataset)
# 生成lstm模型需要的训练集数据
xTest = []
for i in range(timestep, len(xTestDataset)):
    xTest.append(xTestDataset[i - timestep: i])
xTest = np.asarray(xTest)
print(xTest.shape)
yTest = []
for i in range(timestep, len(xTestDataset)):
    yTest.append(yTestDataset[i])
# 反归一化
yTest = scTestDataseY.inverse_transform(X=yTest)
print(yTest.shape)
###############################################################################
# 进行预测
yPredictes = model.predict(x=xTest)
# 反归一化
yPredictes = scTestDataseY.inverse_transform(X=yPredictes)
# print(yPredictes.shape)
###############################################################################
# 对比结果，绘制数据图表，红色是真实数据，蓝色是预测数据
mae, rmse, r2 = [], [], []
for i in range(11):
    plt.plot(yTest[:, i], color='red', label='Real')
    plt.plot(yPredictes[:, i], color='blue', label='Predict')
    plt.title(label='Prediction')
    plt.xlabel(xlabel=columns[i])
    plt.ylabel(ylabel='T')
    plt.legend()
    plt.show()
    # 评估标准： mae, rmse, r2_score
    mae.append(mean_absolute_error(yTest[:, i], yPredictes[:, i]))
    rmse.append(mean_squared_error(yTest[:, i], yPredictes[:, i], squared=False))
    r2.append(r2_score(yTest[:, i], yPredictes[:, i]))

# 得到分数DataFrame类型数据
score_data = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2
}
score = pd.DataFrame(score_data)
score.index = columns
# print(score)
