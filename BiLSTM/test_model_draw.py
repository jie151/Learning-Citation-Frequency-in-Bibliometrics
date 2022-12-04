# 用來計算每個學者各情況下的準確度，可以用來畫圖，另外生成一個scholar_predict_result.txt (在測試檔案中的index, 原本的label, 有沒有預測對)
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
var_epoch_num = 100
var_kFold = 1
var_learning_rate = 0.001
var_dropout = 0.5

# 設定訓練檔與測試檔
#var_testset_file = "../data/2022-11-20/testset_100000.txt"
var_testset_file = "./trainset_50000.txt"
# Create Iterable dataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, each_scholar_vectorLen):
        self.file_path = file_path
        self.each_scholar_vectorLen = each_scholar_vectorLen

    def __iter__(self):
        with open(self.file_path, "r") as file:
            for index, line in enumerate(file):
                if(index % 10000 == 0):
                    print("read file, index: ", index)

                data = line.split()
                # 因為ID不能加，用index取代
                label = [index, int(data[0])]

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
# 2層的list!!!
def save_to_txt(filename, dataList):
    with open(filename, 'a', encoding = 'utf-8') as f:
        for data in dataList:
            for _data in data :
                f.write("%s " % _data)
            f.write("\n")

def remove_exist_file(filename):
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)
        print("remove exist file: ", filename)

def test_model(model, criterion, testset_file):
    model.eval()
    accuracy = 0
    loss = 0
    testset = MyIterableDataset(testset_file, var_input_features_num)
    loader = torch.utils.data.DataLoader(testset, batch_size = var_batch_size)

    scholar_predict_resultFile = "scholar_predict_result.txt"
    remove_exist_file(scholar_predict_resultFile)
    scholar_predict_resultList = []
    count = 0
    with torch.no_grad():
        for data, label in loader:
            count += 1
            if count % 100 == 0 :
                print("test model index: ",count)
                save_to_txt( scholar_predict_resultFile, scholar_predict_resultList)
                scholar_predict_resultList = []
                count = 0

            # 正規化
            data = data_min_max_scaler(data)
            model_out = model(data)
            temp = (model_out >= 0.5).float()
            label_only_list = [] # 只有label的陣列，沒有index
            # 計算準確度
            for t, l in zip(temp, label):
                tempList = [l[0].item(), l[1].item()]
                label_only_list.append(l[1])
                if (t == l[1]):
                    accuracy += 1
                    tempList.extend("Y")
                else:
                    tempList.extend("N")

                scholar_predict_resultList.append(tempList)
            label_only_list = np.array(label_only_list)
            label_only_list = torch.tensor(label_only_list)
            loss += criterion(model_out, label_only_list.float())
        if len(scholar_predict_resultList):
            save_to_txt( scholar_predict_resultFile, scholar_predict_resultList)
            scholar_predict_resultList = []
    testset_num = int(subprocess.getstatusoutput(f"wc -l {var_testset_file}")[1].split()[0])
    loss = loss/testset_num
    accuracy = round(accuracy*100/testset_num, 3)
    print(f"testset\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")

model = BiLSTM()
modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
# Load model
criterion = torch.nn.BCELoss() # mean squared error : input x and target y.
modelName = "model_state_dict.pt"
load_model = BiLSTM()
print("Load model...")
load_model.load_state_dict(torch.load(modelName))
start_time = time.time()
test_model(load_model, criterion, var_testset_file)
execute = time.time() - time.time()
print("test model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))