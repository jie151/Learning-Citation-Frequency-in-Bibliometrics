from    math import ceil
import  numpy
#from    sklearn.preprocessing import MinMaxScaler
import  subprocess
import  time
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torch.utils.data as Data
import  pandas as pd


# parameters for LSTM
nb_lstm_outputs = 1    # 輸出神經元個數
nb_time_steps = 1    # 時間序列的長度
nb_input_vectors =  1000 # 每個輸入序列的向量維度
batch_size = 1200

def generate_dataframe(filename, each_scholar_vectorLen):
    with open(filename, "r") as file:
        dataList = []
        for index, line in enumerate(file):
            if index > 1000:
                break
            data = line.split()

            if len(data) > each_scholar_vectorLen:
                data = data[0:each_scholar_vectorLen + 2]
            else:
                data.extend([0]*(each_scholar_vectorLen - len(data) + 2) )

            dataList.append(data)
    df = pd.DataFrame(dataList)
    return df

dataframe = generate_dataframe("./dataRecord_vector_2_397.txt", nb_input_vectors)
label_df= pd.DataFrame(dataframe[0].astype(int))
dataframe = dataframe.drop(columns=[0, 1])

dataset = dataframe.values
dataset = dataset.astype('float32')

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

trainX = torch.tensor(trainX)
labelX = torch.tensor(labelX)

dataset_train_label = Data.TensorDataset(trainX, labelX)
loader = Data.DataLoader(dataset_train_label, batch_size = 10, shuffle = True)
n_class = nb_input_vectors
n_hidden = 5

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1*2, batch_size, n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size, n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model

model = BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
for epoch in range(1000):
    for x, y in loader:
      pred = model(x)
      loss = criterion(pred, y)
      if (epoch + 1) % 100 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
print("hi")