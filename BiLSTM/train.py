import numpy as np
import subprocess
import time
import torch
from   sklearn.preprocessing import MinMaxScaler
import os

# config
var_model_layer_num = 8         # number of recurrent layers
var_batch_size = 32
var_dataset_size = 1000         # 一次抽多少做kfold
var_input_features_num = 1200   # the number of expected features in the input x
var_hidden_features_num = 512  # the number of features in the hidden state
var_output_dim = 1              # model's output
var_epoch_num = 10
var_kFold = 1
var_learning_rate = 0.001
var_dropout = 0.5

# 設定訓練檔與測試檔
var_trainset_file= "../data/2022-09-21_dupli/trainset_10000.txt"
var_testset_file = "../data/2022-09-21_dupli/testset_10000.txt"

# Create Iterable dataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, each_scholar_vectorLen):
        self.file_path = file_path
        self.each_scholar_vectorLen = each_scholar_vectorLen

    def __iter__(self):
        with open(self.file_path, "r") as file:
            for index, line in enumerate(file):
                if(index % 1000 == 0):
                    print(index)

                data = line.split()
                label = int(data[0])

                # 判斷向量的長度，多的去掉，少的補0
                if len(data) > self.each_scholar_vectorLen + 2:
                    data = data[0:self.each_scholar_vectorLen + 2]
                else:
                    data.extend([0]*(self.each_scholar_vectorLen - len(data) + 2))
                # 取 data[2]之後的值 (data[0] : label, data[1] : ID)
                train = [data[2:]]
                train = np.array(train)
                label = np.array(label)
                train = train.astype("float32")

                train = torch.tensor(train)
                label = torch.tensor(label)
                yield(train, label)

class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size = var_input_features_num,
                                  hidden_size =  var_hidden_features_num,
                                  num_layers = var_model_layer_num,
                                  bidirectional=True,
                                  dropout = var_dropout,
                                  bias = True)
        # fully connected layer
        self.fc = torch.nn.Sequential(torch.nn.Dropout(var_dropout),
                                      torch.nn.Linear(var_hidden_features_num*2, 256),
                                      torch.nn.Linear(256, var_output_dim),
                                      torch.nn.Tanh())
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        # X: [batch_size, seq, feature] =>  input: [seq, batch_size, feature]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)

        hidden_state=torch.randn(var_model_layer_num * 2, batch_size, var_hidden_features_num) # [num_layers * num_directions, batch, hidden_size]
        cell_state  =torch.randn(var_model_layer_num * 2, batch_size, var_hidden_features_num) # [num_layers * num_directions, batch, hidden_size]

        model_out, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))

        out = self.fc(model_out[-1, :, :])

        # shape : (batch_size, -1) -> (batch_size)
        out = out.view(batch_size, -1)
        out = out[:, -1]

        out = self.sigmoid(out)

        return out

def count_label_ratio(datalist):
    update = 0
    for label in datalist:
        if label == 1: update += 1
    return update

def data_min_max_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    nsamples, nx, ny = data.shape
    dim2_data = data.reshape((nsamples, nx*ny))
    data_scaled = scaler.fit_transform(dim2_data)
    data_scaled = data_scaled.reshape(nsamples, nx, ny)
    data_scaled = torch.tensor(data_scaled)

    return data_scaled.float()

def save_to_txt(dataList, filename):
    with open(filename, 'a', encoding = 'utf-8') as f:
        for data in dataList:
            f.write("%s " % data)
        f.write("\n")

def remove_exist_file(filename):
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)
        print("remove exist file: ", filename)

def train_model(model, criterion, optimizer, scheduler, trainset_file):

    for epoch in range(1, var_epoch_num + 1):
        loss_acc_filename = f"train_loss_acc{epoch}.txt"
        remove_exist_file(loss_acc_filename)
        # 批次從檔案中拿出一部分的資料做訓練
        dataset = MyIterableDataset(trainset_file, var_input_features_num)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = var_batch_size)
        print(f"--------\nepoch: {epoch}/{var_epoch_num}, learning rate: {round(scheduler.get_last_lr()[0], 5)}")
        cnt = accuracy_iterator = loss_iterator = 0

        for inputs, labels in dataset_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # 正規化
            inputs = data_min_max_scaler(inputs)
            model.train()
            model_out = model(inputs)

            temp = (model_out >= 0.5).float()
            accuracy_iterator += (temp == labels).sum().item()

            loss  = criterion(model_out, labels.float())
            loss_iterator += loss

            # 反向傳播計算每個參數的梯度值
            loss.backward()

            # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827
            optimizer.step()

            cnt += len(labels)

            if  cnt > 500:
                acc = accuracy_iterator/cnt
                los = loss_iterator/cnt
                dataList = [loss_iterator.item(), accuracy_iterator, cnt]
                save_to_txt(dataList, loss_acc_filename)
                cnt = accuracy_iterator = loss_iterator = 0
                print(f"iterator: loss: {'{:.3f}'.format(los)}, acc: {round(acc, 3)}")

        if(cnt != 0):
            dataList = [loss_iterator.item(), accuracy_iterator, cnt]
            save_to_txt(dataList, loss_acc_filename)

        with open(loss_acc_filename, "r") as file:
            accuracy_epoch = loss_epoch = dataSize = 0
            for line in file:
                line = line.split()
                accuracy_epoch += int(line[1])
                loss_epoch += float(line[0])
                dataSize += int(line[2])

        print(f"*****loss: {'{:.3f}'.format(loss_epoch/dataSize)}, accuracy: {round(accuracy_epoch/dataSize,3)}, dataSize: {dataSize}")

def test_model(model, criterion, testset_file):
    model.eval()
    accuracy_all = 0
    loss_all = 0
    cnt = 0
    accuracy = 0
    loss = 0
    testset = MyIterableDataset(testset_file, var_input_features_num)
    loader = torch.utils.data.DataLoader(testset, batch_size = var_batch_size)
    with torch.no_grad():
        for data, label in loader:
            cnt += len(label)
            # 正規化
            data = data_min_max_scaler(data)
            model_out = model(data)

            temp = (model_out >= 0.5).float()
            accuracy += (temp == label).sum().item()
            loss     += criterion(model_out, label.float())
            if(cnt > 500):
                accuracy_all = (accuracy/cnt + accuracy_all) / 2
                loss_all = (loss/cnt + loss_all) / 2
                cnt = 0
                accuracy = 0
                loss = 0
    if(cnt != 0):
        accuracy_all = (accuracy/cnt + accuracy_all) / 2
        loss_all = (loss/cnt + loss_all) / 2
    print(f"testset\nLoss: { '{:.3f}'.format(loss_all) }, Accuracy: {round(accuracy_all, 3)}%")

start_time = time.time()

model = BiLSTM()
criterion = torch.nn.BCELoss() # mean squared error : input x and target y.
optimizer = torch.optim.Adam(model.parameters(), lr=var_learning_rate) #0.01-0.001 lr = 0.001
optimizer_ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
train_model(model, criterion, optimizer , optimizer_ExpLR, var_trainset_file)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
print(f"save model's parameter: {modelName}")

# Load model
load_model = BiLSTM()
print("Load model...")
load_model.load_state_dict(torch.load(modelName))
test_model(load_model, criterion, var_testset_file)