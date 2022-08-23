from tensorflow import keras
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy

# parameters for LSTM
nb_lstm_outputs = 1    # 輸出神經元個數
nb_time_steps = 1    # 時間序列的長度
nb_input_vectors =  1000 # 每個輸入序列的向量維度

def generate_dataframe(filename, each_scholar_vectorLen):
    with open(filename, "r") as file:
        dataList = []
        for index, line in enumerate(file):
            data = line.split()

            if len(data) > each_scholar_vectorLen:
                data = data[0:each_scholar_vectorLen + 2]
            else:
                data.extend([0]*(each_scholar_vectorLen - len(data) + 2) )

            dataList.append(data)
    df = pd.DataFrame(dataList)
    return df

dataframe = generate_dataframe("./dataRecord_vector_2_397.txt", nb_input_vectors)
label_df= pd.DataFrame(dataframe[0].astype(int)) # label: 例如:第二筆，存第三筆與第二筆資料的引用次是否不同
dataframe = dataframe.drop(columns=[0, 1]) # 去掉 label(下一次是否跟這次不同), ID

dataset = dataframe.values
dataset = dataset.astype('float32')
# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 2/3 資料為訓練資料， 1/3 資料為測試資料
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX = numpy.reshape(train, (train.shape[0], 1, train.shape[1]))
testX = numpy.reshape(test, (test.shape[0], 1, test.shape[1]))

#label_df = pd.read_csv('dataRecord_vector_2.txt', sep=" ", header = None, usecols = [0])
labelX = numpy.array(label_df[0][:train_size])
labeltest = numpy.array(label_df[0][train_size:])

print(f"trainX: {trainX.shape},labelX: {labelX.shape}")
print(f"testX: {testX.shape},labelX: {labeltest.shape}")
print(labelX)

# building model
model = Sequential()

model.add(
    Bidirectional(
        LSTM(units=64, return_sequences=True),
        input_shape=(nb_time_steps, nb_input_vectors,)
    )
)
model.add(Bidirectional(LSTM(units=32)))

model.add(Dense(16, activation="relu"))

model.add(Dense(nb_lstm_outputs, activation='relu'))

# compile:loss, optimizer, metrics
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit( trainX, labelX, epochs=20, batch_size=10, verbose=1)

#model.save("my_model.h5")

score = model.evaluate(testX, labeltest, batch_size=10, verbose=1)
print(score)