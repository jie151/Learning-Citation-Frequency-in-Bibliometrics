import numpy as np
import subprocess
import time
import torch
from   sklearn.model_selection import StratifiedKFold
from   sklearn.preprocessing import MinMaxScaler
from   module.save_to_txt import save_to_txt

# config
batch_size = 10
input_features = 800   # the number of expected features in the input x
hidden_feature_dim = 512 # the number of features in the hidden state
model_layer_num = 8     # number of recurrent layers
output_dim = 1          # model's output
epochs_num = 10
kFolds_num = 5
learningRate = 0.01
subDataset_size = 10
dropout = 0.5
# 設定訓練檔與測試檔
trainFile= "./random_balance_10.txt"
testFile = "./random_balance_10.txt"
trainset_num = subprocess.getstatusoutput(f"wc -l {trainFile}")[1].split()[0]
testset_num = subprocess.getstatusoutput(f"wc -l {testFile}")[1].split()[0]
print(f"train: {trainset_num}, test: {testset_num} ")
train_arg_list =  {"path": trainFile, "each_scholar_vectorLen": input_features}
test_arg_list = {"path": testFile, "each_scholar_vectorLen": input_features}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

# Create Iterable dataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, arg_list):
        self.filePath = arg_list["path"]
        self.each_scholar_vectorLen = arg_list["each_scholar_vectorLen"]

    def __iter__(self):
        with open(self.filePath, "r") as file:
            for index, line in enumerate(file):

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

                # 正規化
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler.fit_transform(train.reshape(self.each_scholar_vectorLen, -1))
                train = train_scaled.reshape(-1, self.each_scholar_vectorLen)

                #print(f"{data[1]} train.shape: {train.shape}, label.shape: {label.shape} ")
                train = torch.tensor(train)
                label = torch.tensor(label)
                yield(train, label)

class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_features,
                                  hidden_size =  hidden_feature_dim,
                                  num_layers = model_layer_num,
                                  bidirectional=True)
                                  #bias = False) # input, output: (seq, batch_size, feature)
        # fully connected layer
        self.fc =torch.nn.Linear(hidden_feature_dim*2, output_dim)
        self.Dropout = torch.nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, X):
        # X: [batch_size, seq, feature]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [seq, batch_size, feature]

        hidden_state=torch.randn(model_layer_num * 2, batch_size, hidden_feature_dim) # [num_layers * num_directions, batch, hidden_size]
        cell_state  =torch.randn(model_layer_num * 2, batch_size, hidden_feature_dim) # [num_layers * num_directions, batch, hidden_size]

        model_out, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))

        out = self.fc(model_out[-1,:,:])
        out = self.tanh(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out

# 計算準確率
def count_accuracy(outputs, labels):
    count = 0
    for out, lab in zip(outputs, labels):
        if (out >= 0.5 and lab == 1) or (out < 0.5 and lab == 0):
            count += 1
    return count

def train_model(model, epochs_num, batch_size, criterion, optimizer):
    # Defind the k-fold cross validator
    kFold = StratifiedKFold(n_splits = kFolds_num, random_state=202, shuffle=True)

    for epoch in range(epochs_num):
        # 批次從檔案中拿出一部分的資料作為dataset
        dataset = MyIterableDataset(train_arg_list)
        loader = torch.utils.data.DataLoader(dataset, batch_size = subDataset_size, shuffle=False)

        for data_subset, label_subset in loader:
            data_label_subset = MyDataset(data_subset, label_subset)
            all_fold_accuracy = {}

            print(f"--------\nepoch: {epoch+1}/{epochs_num}")
            for fold, (train_indexs, valid_indexs) in enumerate(kFold.split(data_subset, label_subset)):
                print(f"train/valid size: {len(train_indexs)}/{len(valid_indexs)}")

                # Sample elements randomly from a given list of indexs, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indexs)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indexs)

                # Define data loaders for training and validating data in this fold
                trainLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = batch_size, sampler=train_subsampler)#, shuffle=False)
                validLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = batch_size, sampler=valid_subsampler) #shuffle=False)

                # Set loss, accuracy value
                loss_fold     = 0
                accuracy_fold = 0

                # Iterate over the DataLoader for training data
                for inputs, labels in trainLoader:
                    #print(inputs)
                    optimizer.zero_grad() # Zero the gradients

                    model.train()
                    model_out = model(inputs)

                    accuracy_fold += count_accuracy(model_out, labels.float())

                    loss = criterion(model_out, labels.float())
                    loss_fold += loss

                    loss.backward() # 反向傳播計算每個參數的梯度值
                    optimizer.step() # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827

                # 一個fold的準確率與loss
                accuracy_fold = round(accuracy_fold*100/len(train_indexs), 3)
                loss_fold = loss_fold/len(train_indexs)

                print(f"fold {fold+1}/{kFolds_num}, Loss: { '{:.3f}'.format(loss_fold) }, Accuracy: {accuracy_fold}%")

                # validation theis fold
                with torch.no_grad():
                    correct = 0
                    for valid_inputs, label in validLoader:
                        model_out = model(valid_inputs)

                        correct += count_accuracy(model_out, label.float())
                    all_fold_accuracy[fold] = 100.0 * (correct/len(valid_indexs))
            print(f"********\nKFold cross validation results {kFolds_num} folds")
            sum = 0.0
            for key, value in all_fold_accuracy.items():
                print(f"fold {key+1}: {round(value, 3)} %")
                sum += value
            print(f"average: {round(sum/len(all_fold_accuracy.items()),3)} %\n")

def test_model(model, criterion):
    model.eval()
    accuracy = 0
    loss = 0
    testset = MyIterableDataset(test_arg_list)
    loader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
    with torch.no_grad():
        for data, label in loader:
            model_out = model(data)
            accuracy += count_accuracy(model_out, label)
            # 計算loss
            loss = criterion(model_out, label.float())

    loss = loss/int(testset_num)
    accuracy = round(accuracy*100/int(testset_num), 3)
    print(f"test\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")

start_time = time.time()

model = BiLSTM()
criterion = torch.nn.BCELoss() # mean squared error : input x and target y.
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.01-0.001 lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate) #0.01-0.001 lr = 0.001
train_model(model, epochs_num, batch_size, criterion, optimizer)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
print(f"save model's parameter: {modelName}")

# Load model
load_model = BiLSTM()
load_model.load_state_dict(torch.load(modelName))
test_model(load_model, criterion)