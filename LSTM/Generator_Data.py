from LSTM.Data_Preparing import DataPreparing
if __name__ == '__main__':
    raw_data = DataPreparing(536)
    train_x1, train_x2 = raw_data.get_data()
    # raw_data.get_trace_time()
    with open('./application1_train_x.txt', 'w') as f:
        f.write(str(train_x1))
    with open('./application2_train_x.txt', 'w') as f:
        f.write(str(train_x2))