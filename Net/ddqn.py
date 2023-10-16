# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


import tensorflow as tf
from PER.memory_buffer import MemoryBuffer
from Evaluation.LossRecord import LossHistory

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.input_shape = state_space
        self.action_space = action_space
        self.memory = MemoryBuffer(5000)
        self.gamma = 0.9    # discount rate
        self.alpha = 0.1  # learning rate
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        # self.current_steps = 1
        # 记录loss
        self.history = LossHistory()
        # self.recorder = LossRecord('./losses/w1=' + str(w1) + '-w2=' + str(w2) + '-th_under=' + str(th_under) + '-loss.txt')

    def huber_loss(self, y_true, y_pred):
        # if not K.is_keras_tensor(y_pred):
        #     y_pred = K.constant(y_pred)
        # y_true = K.cast(y_true, y_pred.dtype)
        # return K.mean(K.square(y_pred - y_true), axis=(1, 2))
        return K.sum(K.mean(K.square(y_pred - y_true), axis=-1), axis=-1)

    #  构建网络
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_shape=self.input_shape, activation='relu'))
        model.add(Dense(128))
        model.add(Dense(128))
        model.add(Dense(self.action_space))

        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    #  更新目标网络
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    #  存储样本
    def memorize(self, state, action, reward, next_state, done):
        q_s = self.model.predict(state.reshape((1,) + state.shape))[0]  # shape(3,  67)
        q_ns = self.target_model.predict(next_state.reshape((1,) + next_state.shape))[0]  # shape(3,  67)
        next_best_action = np.argmax(self.model.predict(next_state.reshape((1,) + next_state.shape)), axis=-1)[0]  # shape(3, )
        q_s_a = np.zeros((state.shape[0],))  # shape(3, )
        q_ns_na = np.zeros((state.shape[0],))
        for i in range(state.shape[0]):
            q_s_a[i] = q_s[i, action[i]-1]
            if done == 1:
                q_ns_na[i] = reward
            else:
                q_ns_na[i] = reward + self.gamma * q_ns[i, next_best_action[i]]
        td_error = abs(np.sum(q_s_a)-np.sum(q_ns_na))
        self.memory.memorize(state, action, reward, next_state, done, td_error)

        # last version
        # q_val = self.model.predict(state.reshape((1,)+state.shape))[0]
        # q_val_t = self.target_model.predict(next_state.reshape((1,)+next_state.shape))[0]
        # next_best_action = np.argmax(self.model.predict(next_state.reshape((1,)+next_state.shape)), axis=-1)[0]
        # old_val = []
        # new_val = []
        # for i in range(state.shape[0]):
        #     old_val.append(q_val[i, action[i]])
        #     if done == 1:
        #         new_val.append(reward)
        #     else:
        #         new_val.append(reward + self.gamma * q_val_t[i, next_best_action[i]])
        # td_error = np.sum(abs(np.array(new_val) - np.array(old_val)))
        # print('------------------------memorize-------------------------')
        # self.memory.memorize(state, action, reward, next_state, done, td_error)

    #  经验回放
    def replay(self, batch_size):
        s, a, r, new_s, d, idx = self.memory.sample_batch(batch_size)
        q_s = self.model.predict(s)  # shape(batch_size, 3, 67)
        q_ns = self.target_model.predict(new_s)
        argmax_action = np.argmax(self.model.predict(new_s), axis=-1)  # shape(batch_size, 67)
        for i in range(s.shape[0]):
            q_s_a = np.zeros((s.shape[1],))
            q_ns_na = np.zeros((s.shape[1],))
            for j in range(s.shape[1]):
                q_s_a[j] = q_s[i, j, a[i, j]-1]
                if d[i] == 1:
                    q_ns_na[j] = r[i]
                else:
                    q_ns_na[j] = r[i] + self.gamma * q_ns[i, j, argmax_action[i, j]]
                q_s[i, j, a[i, j]-1] = q_ns_na[j]
            td_error = abs(np.sum(q_s_a) - np.sum(q_ns_na))
            self.memory.update(idx[i], td_error)
        self.model.fit(s, q_s, callbacks=[self.history])

        # last version
        # Apply Bellman Equation on batch samples to train our DDQN
        # q = self.model.predict(s)
        # next_q = self.model.predict(new_s)
        # q_targ = self.target_model.predict(new_s)
        #
        # for i in range(s.shape[0]):
        #     q_i = q[i]
        #     next_q_i = next_q[i]
        #     q_targ_i = q_targ[i]
        #     a_i = a[i]
        #     r_i = r[i]
        #     old_q_val = []
        #     new_q_val = []
        #     for j in range(s.shape[1]):
        #         old_q_val.append(q_i[j, a_i[j]])
        #         if d[i] == 1:
        #             q_i[j, a_i[j]] = r_i
        #         else:
        #             next_best_action = np.argmax(next_q_i, axis=1)
        #             q_i[j, a_i[j]] = r_i + self.gamma * q_targ_i[j, next_best_action[j]]
        #         new_q_val.append(q_i[j, a_i[j]])
        #
        #     self.memory.update(idx[i], np.sum(abs(np.array(old_q_val)-np.array(new_q_val))))
        # # Train on batch
        # self.model.fit(s, q)
        # 记录losses
        # self.recorder.record(self.history.losses)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


