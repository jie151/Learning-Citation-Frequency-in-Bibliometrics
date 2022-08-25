from math import ceil
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
import subprocess

# parameters for LSTM
nb_lstm_outputs = 1    # 輸出神經元個數
nb_time_steps = 1    # 時間序列的長度
nb_input_vectors =  1000 # 每個輸入序列的向量維度
batch_size = 400

def generate_arrays_from_file(path, batch_size, each_scholar_vectorLen):
    cnt = 0
    data_batch = []
    label_batch = []

    while 1:
        f = open(path)

        for index, line in enumerate(f):
            data = line.split()

            label_batch.append(int(data[0]))
            # 判斷向量的長度，多的去掉，少的補0
            if len(data) > each_scholar_vectorLen + 2:
                data = data[0:each_scholar_vectorLen + 2]
            else:
                data.extend([0]*(each_scholar_vectorLen - len(data) + 2))
            data_batch.append(data[2:]) # data[0] : label, data[1] : ID
            cnt += 1
            #print(f"{index}, cnt: {cnt} ")
            if (cnt == batch_size):
                cnt = 0
                dataset = np.array(data_batch)
                dataset = dataset.astype('float32')
                # 正規化，使資料值介於[0, 1]
                #scaler = MinMaxScaler(feature_range=(0, 1))
                #dataset = scaler.fit_transform(dataset)
                train = np.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))
                label = np.array(label_batch)
                #print(f"\nindex: {index} trainX: {train.shape},labelX: {label.shape}")
                yield(train, label)
                data_batch = []
                label_batch = []
        f.close()

# building model
model = Sequential()

model.add(
    Bidirectional(
        LSTM(units=64, return_sequences=True),#LSTM(units=nb_lstm_outputs, return_sequences=True),
        input_shape=(nb_time_steps, nb_input_vectors,)
    )
)
model.add(Bidirectional(LSTM(units=32)))

model.add(Dense(16, activation="relu"))

model.add(Dense(nb_lstm_outputs, activation='relu'))

# compile:loss, optimizer, metrics
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

filename = "./dataRecord_vector_2.txt"
fileLine = subprocess.getstatusoutput(f"wc -l {filename}")[1].split()[0]

step = ceil(int(fileLine) / batch_size)

start_time = time.time()
model.fit(generate_arrays_from_file(filename, batch_size, nb_input_vectors), steps_per_epoch = step, epochs=10, verbose=2)
execute = (time.time() - start_time)
print("model fit : ",time.strftime("%H:%M:%S", time.gmtime(execute)))
print("the number of data: ", fileLine)



