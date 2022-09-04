from    math import ceil
import  numpy as np
#from    sklearn.preprocessing import MinMaxScaler
import  subprocess
import  time
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torch.utils.data as Data

class MyIterableDataset(Data.IterableDataset):

    def __init__(self, arg_list):
        self.filePath = arg_list["path"]
        self.each_scholar_vectorLen = arg_list["each_scholar_vectorLen"]
        self.batch_size = arg_list["batch_size"]

    def __iter__(self):
        train  = []
        label  = []

        with open(self.filePath, "r") as file:

            for index, line in enumerate(file):

                data = line.split()
                label = int(data[0])

                # 判斷向量的長度，多的去掉，少的補0
                if len(data) > self.each_scholar_vectorLen + 2:
                    data = data[0:self.each_scholar_vectorLen + 2]
                else:
                    data.extend([0]*(self.each_scholar_vectorLen - len(data) + 2))

                train = [data[2:]] # data[0] : label, data[1] : ID

                train = np.array(train)
                label = np.array(label)

                train = train.astype("float32")

                #print(f"{data[1]} train.shape: {train.shape}, label.shape: {label.shape} ")

                train = torch.tensor(train)
                label = torch.tensor(label)
                yield(train, label)


each_scholar_vectorLen = 1000
batch_size = 10
arg_list =  {"path": "dataRecord_vector_2_397.txt", "each_scholar_vectorLen": each_scholar_vectorLen, "batch_size": batch_size}


input_features = 1000 # the number of expected features in the input x
hidden_feature_dim = 50 # the number of features in the hidden state
lstm_layer_num = 3 # number of recurrent layers
output_dim = 1

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size = input_features, hidden_size =  hidden_feature_dim, num_layers = lstm_layer_num,
                            bidirectional=True) # input, output: ( seq, batch_size, feature)

        # fully connected layer
        self.fc = nn.Linear(hidden_feature_dim*2, output_dim)

    def forward(self, X):
        # X: [batch_size, seq, feature]
        batch_size = X.shape[0]

        input = X.transpose(0, 1)  # input : [seq, batch_size, feature]

        hidden_state=torch.randn(lstm_layer_num * 2, batch_size, hidden_feature_dim)    # [num_layers * num_directions, batch, hidden_size]
        cell_state  =torch.randn(lstm_layer_num * 2, batch_size, hidden_feature_dim)    # [num_layers * num_directions, batch, hidden_size]

        outputs, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))

        outputs = outputs[-1]  # [batch_size, hidden_feature_dim * 2]

        model = self.fc(outputs)  # model : [batch_size, seq]

        return model
model = BiLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(100):
    train_set = MyIterableDataset(arg_list)
    loader = Data.DataLoader(train_set, batch_size = 10)

    for x, y in loader:

        pred = model(x)

        #pred = pred.to(torch.float32)

        y = y.to(torch.float32)

        #for x_, y_ in zip(pred, y):
           #print(x_, " ", y_)

        loss = criterion(pred, y)

        #if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad() # 將梯度歸0
        loss.backward() # 反向傳播計算每個參數的梯度值
        optimizer.step() # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827


