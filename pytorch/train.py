from    math import ceil
import  numpy as np
#from    sklearn.preprocessing import MinMaxScaler
import  subprocess
import  time
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torch.utils.data as Data
from module.save_to_txt import save_to_txt

# config
batch_size = 100
input_features = 1200   # the number of expected features in the input x
hidden_feature_dim = 32 # the number of features in the hidden state
lstm_layer_num = 4      # number of recurrent layers
output_dim = 1          # model's output
num_epochs = 10
trainFile= "./dataRecord_vector_2_397.txt"
testFile = "./dataRecord_vector_2_397.txt"
trainset_num = subprocess.getstatusoutput(f"wc -l {trainFile}")[1].split()[0]
testset_num = subprocess.getstatusoutput(f"wc -l {trainFile}")[1].split()[0]
print(f"train: {trainset_num}, test: {testset_num} ")
train_arg_list =  {"path": trainFile, "each_scholar_vectorLen": input_features, "batch_size": batch_size}
test_arg_list = {"path": testFile, "each_scholar_vectorLen": input_features, "batch_size": batch_size}

# 創建iterable dateset
class MyIterableDataset(Data.IterableDataset):

    def __init__(self, arg_list):
        self.filePath = arg_list["path"]
        self.each_scholar_vectorLen = arg_list["each_scholar_vectorLen"]
        self.batch_size = arg_list["batch_size"]

    def __iter__(self):
        with open(self.filePath, "r") as file:
            for index, line in enumerate(file):
                if (index % 5000 == 0):
                    print(index)
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

        model_out, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))
        #print(model_out.shape) (1, 10:batch, 100:feature)
        output = self.fc(model_out[-1,:,:])
        #model_out = model_out[-1]  # [batch_size, hidden_feature_dim * 2]
        #output = self.fc(model_out)  # model : [batch_size, seq]
        return output

model = BiLSTM()
criterion = nn.MSELoss() # mean squared error : input x and target y.
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        train_set = MyIterableDataset(train_arg_list)
        loader = Data.DataLoader(train_set, batch_size = 10) #shuffle=False)
        accuracy = 0

        for data, label in loader:

            model_output = model(data)
            #if model_output > 0.5, result = 1, else: 0
            result = (model_output > 0.5).float()
            label = label.to(torch.float32)

            # 將tensor -> list, 存預測值與label
            x = result.tolist()
            y = label.tolist()

            for _model_out, _label, _result in zip(model_output, label, result):
                #print(_model_out, _result, _label)
                if (_label == _result):
                    accuracy += 1

            x_y_list = [tmpX + [tmpY] for tmpX, tmpY in zip(x, y)]
            #save_to_txt("myfile.txt", x_y_list)
            # 計算loss
            loss = criterion(model_output, label)

            optimizer.zero_grad() # 將梯度歸0
            loss.backward() # 反向傳播計算每個參數的梯度值
            optimizer.step() # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827
            #print(loss)
        epoch_accuracy = round(accuracy*100/int(trainset_num), 3)
        print(accuracy)
        print(f"Eopch {epoch+1}/{num_epochs}, Loss: { '{:.3f}'.format(loss) }, Accuracy: {epoch_accuracy}%")

def test(model):
    model.eval()
    accuracy = 0

    testset = MyIterableDataset(test_arg_list)
    loader = Data.DataLoader(testset, batch_size = 10, shuffle=False)
    with torch.no_grad():
        for data, label in loader:
            model_out = model(data)
            #if model_output > 0.5, result = 1, else: 0
            result = (model_out > 0.5).float()
            label = label.to(torch.float32)

            # 將tensor -> list, 存預測值與label
            x = result.tolist()
            y = label.tolist()

            for _model_out, _label, _result in zip(model_out, label, result):
                #print(_model_out, _result, _label)
                if (_label == _result):
                    accuracy += 1

            x_y_list = [tmpX + [tmpY] for tmpX, tmpY in zip(x, y)]
            #save_to_txt("myfile.txt", x_y_list)
            # 計算loss
            loss = criterion(model_out, label)
    print (accuracy)
    accuracy = round(accuracy*100/int(testset_num), 3)
    print(f"test\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")

start_time = time.time()
train(num_epochs)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
print(f"save model's parameter: {modelName}")

# Load model
load_model = BiLSTM()
load_model.load_state_dict(torch.load(modelName))

test(load_model)