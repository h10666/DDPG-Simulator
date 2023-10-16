import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Input, Activation, Dense, TimeDistributed, LSTM
from keras.optimizers import Adam
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class UserDistributionPredict:
    def __init__(self):
        self.sample_num = 26000
        self.test_sample = 2000
        self.time_steps = 15
        self.input_size = 9
        self.epochs = 5000
        self.batch_size = 128
        self.index_start = 0
        self.output_size = 9
        self.cell_size = 128
        self.lr = 5e-4

        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

        self.sc_train_x = None
        self.sc_train_y = None
        self.sc_test_x = None
        self.sc_test_y = None

        self.model = self.create_model()

    def sc_fit_transform(self, raw_data):
        # 将所有数据归一化为0-1的范围
        sc = MinMaxScaler(feature_range=(0, 1))
        dataset_transform = sc.fit_transform(X=raw_data)
        # 归一化后的数据
        return sc, np.array(dataset_transform)

    def get_data(self):
        with open('application2_train_x.txt', 'r') as f:
            raw_data = np.asarray(ast.literal_eval(f.read()))
            train_x = raw_data[0:self.sample_num]
            train_y = raw_data[1:self.sample_num+1]
            test_x = raw_data[self.sample_num: self.sample_num+self.test_sample]
            test_y = raw_data[self.sample_num+1: self.sample_num+self.test_sample+1]

            self.sc_train_x, train_x = self.sc_fit_transform(train_x)
            self.sc_train_y, train_y = self.sc_fit_transform(train_y)
            self.sc_test_x, test_x = self.sc_fit_transform(test_x)
            self.sc_test_y, test_y = self.sc_fit_transform(test_y)

            for i in range(self.time_steps, self.sample_num):
                self.X_train.append(train_x[i-self.time_steps: i])
                self.Y_train.append(train_y[i])
            for i in range(self.time_steps, self.test_sample):
                self.X_test.append(test_x[i-self.time_steps: i])
                self.Y_test.append(test_y[i])

            self.X_train = np.asarray(self.X_train)  # (14000, TIME_STEPS, 64)
            self.Y_train = np.asarray(self.Y_train)  # (14000, 64)
            self.X_test = np.asarray(self.X_test)  # (300, TIME_STEPS, 64)
            self.Y_test = np.asarray(self.Y_test)  # (300, 64)
            # print(self.X_train.shape)
            # print(self.Y_train.shape)
            # print(self.X_test.shape)
            # print(self.Y_test.shape)

    def create_model(self):
        # model = Sequential()
        # model.add(LSTM(self.cell_size, input_shape=(self.time_steps, self.input_size)))
        # model.add(Dense(self.output_size))
        # model.compile(loss='mse', optimizer=Adam(self.lr), metrics=['accuracy'])
        # return model
        return load_model('../LSTM/model1.h5')

    def train(self):
        self.get_data()
        # for i in range(40000):
        #     self.X_batch = self.X_train[self.index_start:self.index_start + self.batch_size, :, :]
        #     self.Y_batch = self.Y_train[self.index_start:self.index_start + self.batch_size, :]
        #     self.index_start += self.batch_size
        #     self.model.fit(self.X_batch, self.Y_batch)
        #     if self.index_start >= self.X_train.shape[0]:
        #         self.index_start = 0
        #     if i % 1000 == 0:
        #         self.lr = self.lr*0.1
        #         loss, acc = self.model.evaluate(self.X_test, self.Y_test)
        #         print('loss:' + str(loss))
        #         print('accuracy:' + str(acc))

        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)
        self.model.save('./model1.h5')

    def predict(self):
        self.get_data()
        self.Y_test = self.sc_test_y.inverse_transform(X=self.Y_test)
        predict_y = self.model.predict(self.X_test)
        predict_y = np.ceil(abs(self.sc_test_y.inverse_transform(X=predict_y)))
        mae, rmse, r2 = [], [], []
        for i in range(self.output_size):
            plt.plot(self.Y_test[:, i], color='red', label='Real')
            plt.plot(predict_y[:, i], color='blue', label='Predict')
            plt.title(label='edge server' + str(i) + ' prediction')
            plt.xlabel(xlabel='Time')
            plt.ylabel(ylabel='users')
            plt.legend()
            plt.savefig('./app1-results/edge server' + str(i) + ' prediction.png')
            plt.show()
            # 评估标准： mae, rmse, r2_score
            mae.append(mean_absolute_error(self.Y_test[:, i], predict_y[:, i]))
            rmse.append(mean_squared_error(self.Y_test[:, i], predict_y[:, i], squared=False))
            r2.append(r2_score(self.Y_test[:, i], predict_y[:, i]))
        print('mae')
        print(mae)
        print('rmse')
        print(rmse)
        print('r2')
        print(r2)




if __name__ == '__main__':
    user_predict = UserDistributionPredict()
    # user_predict.train()
    user_predict.predict()
