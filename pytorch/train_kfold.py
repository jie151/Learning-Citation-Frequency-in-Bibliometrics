import numpy as np
import subprocess
import time
import torch
from   sklearn.model_selection import StratifiedKFold
from   sklearn.preprocessing import MinMaxScaler

# config
modelLayerNum = 6        # number of recurrent layers
batchSize = 32
datasetSize = 2000         # 一次抽多少做kfold
inputFeatures = 1200     # the number of expected features in the input x
hiddenFeatureDim = 1024  # the number of features in the hidden state
outputDim = 1            # model's output
epochNum = 10
kFoldNum = 5
learningRate = 0.001
dropout = 0.5

# 設定訓練檔與測試檔
trainFile= "./random_balance_10.txt"
testFile = "./random_balance_10.txt"
#trainFile_num = subprocess.getoutput(f"wc -l {trainFile}")[1].split()[0]
testsetNum = subprocess.getstatusoutput(f"wc -l {testFile}")[1].split()[0]
train_arg_list =  {"path": trainFile, "each_scholar_vectorLen": inputFeatures}
test_arg_list = {"path": testFile, "each_scholar_vectorLen": inputFeatures}

# 將向量與label再合成dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data   = data_tensor
        self.label = label_tensor
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return self.data.size(0)

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

                train = torch.tensor(train)
                label = torch.tensor(label)
                yield(train, label)

class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size = inputFeatures,
                                  hidden_size =  hiddenFeatureDim,
                                  num_layers = modelLayerNum,
                                  bidirectional=True,
                                  dropout = dropout,
                                  bias = True) # input, output: (seq, batch_size, feature)
        # fully connected layer
        self.fc = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                      torch.nn.Linear(hiddenFeatureDim*2, 256),
                                      torch.nn.Linear(256, outputDim),
                                      torch.nn.Tanh()) # Tanh
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        # X: [batch_size, seq, feature] =>  input: [seq, batch_size, feature]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)

        hidden_state=torch.randn(modelLayerNum * 2, batch_size, hiddenFeatureDim) # [num_layers * num_directions, batch, hidden_size]
        cell_state  =torch.randn(modelLayerNum * 2, batch_size, hiddenFeatureDim) # [num_layers * num_directions, batch, hidden_size]

        model_out, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))

        out = self.fc(model_out[-1, :, :])

        # shape : (batch_size, -1) -> (batch_size)
        out = out.view(batch_size, -1)
        out = out[:, -1]

        out = self.sigmoid(out)

        return out

# 計算準確率
def count_accuracy(outputs, labels):
    count = 0
    outputs = (outputs >= 0.5).float()
    for out, lab in zip(outputs, labels):
        if (out == lab):
            count += 1
    return count

def train_model(model, criterion, optimizer):
    # Defind the k-fold cross validator
    kFold = StratifiedKFold(n_splits = kFoldNum, random_state=202, shuffle=True)

    for epoch in range(epochNum):
        # 批次從檔案中拿出一部分的資料作為dataset
        dataset = MyIterableDataset(train_arg_list)
        loader = torch.utils.data.DataLoader(dataset, batch_size = datasetSize, shuffle=False)

        for data_subset, label_subset in loader:
            # 正規化
            scaler = MinMaxScaler(feature_range=(0, 1))
            nsamples, nx, ny = data_subset.shape

            dim2_inputs = data_subset.reshape((nsamples, nx*ny))
            inputs_scaled = scaler.fit_transform(dim2_inputs)
            dim3_inputs = inputs_scaled.reshape(nsamples, nx, ny)
            dim3_inputs = torch.tensor(dim3_inputs)
            data_subset = dim3_inputs.float()


            data_label_subset = MyDataset(data_subset, label_subset)
            all_fold_accuracy = {}

            print(f"--------\nepoch: {epoch+1}/{epochNum}")
            for fold, (train_indexs, valid_indexs) in enumerate(kFold.split(data_subset, label_subset)):
                print(f"train/valid size: {len(train_indexs)}/{len(valid_indexs)}")

                # Sample elements randomly from a given list of indexs, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indexs)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indexs)

                # Define data loaders for training and validating data in this fold
                trainLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = batchSize, sampler=train_subsampler)#, shuffle=False)
                validLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = batchSize, sampler=valid_subsampler) #shuffle=False)

                # Set loss, accuracy value
                loss_fold     = 0
                accuracy_fold = 0
                count = 0
                # Iterate over the DataLoader for training data
                for inputs, labels in trainLoader:
                    count += 1
                    #print(inputs)
                    optimizer.zero_grad() # Zero the gradients

                    model.train()

                    model_out = model(inputs)

                    # 計算準確率
                    tempOut = (model_out >= 0.5).float()
                    accuracy_fold += (tempOut == labels).sum().item()

                    loss = criterion(model_out, labels.float())

                    loss_fold += loss

                    loss.backward() # 反向傳播計算每個參數的梯度值
                    optimizer.step() # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827

                # 一個fold的準確率與loss
                accuracy_fold = round(accuracy_fold*100/len(train_indexs), 3)
                loss_fold = loss_fold/count


                print(f"fold {fold+1}/{kFoldNum}, Loss: { '{:.3f}'.format(loss_fold) }, Accuracy: {accuracy_fold}%")

                # validation theis fold
                with torch.no_grad():
                    correct = 0
                    for valid_inputs, label in validLoader:
                        model_out = model(valid_inputs)

                        correct += count_accuracy(model_out, label.float())
                    all_fold_accuracy[fold] = 100.0 * (correct/len(valid_indexs))
            print(f"********\nKFold cross validation results {kFoldNum} folds")
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
    loader = torch.utils.data.DataLoader(testset, batch_size = batchSize)
    with torch.no_grad():
        count = 0
        for data, label in loader:
            # 正規化
            scaler = MinMaxScaler(feature_range=(0, 1))
            nsamples, nx, ny = data.shape

            dim2_inputs = data.reshape((nsamples, nx*ny))
            inputs_scaled = scaler.fit_transform(dim2_inputs)
            dim3_inputs = inputs_scaled.reshape(nsamples, nx, ny)
            dim3_inputs = torch.tensor(dim3_inputs)
            data = dim3_inputs.float()

            count += 1
            model_out = model(data)
            temp = (model_out >= 0.5).float()
            #print(temp)
            accuracy += (temp == label).sum().item()
            # 計算loss
            loss = criterion(model_out, label.float())

    loss = loss/count
    print(accuracy,int(testsetNum))
    accuracy = round(accuracy*100/int(testsetNum), 3)
    print(f"test\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")

start_time = time.time()

model = BiLSTM()
criterion = torch.nn.BCELoss() # mean squared error : input x and target y.
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #0.01-0.001 lr = 0.001
train_model(model, criterion, optimizer)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
print(f"save model's parameter: {modelName}")

# Load model
load_model = BiLSTM()
print("Load model...")
load_model.load_state_dict(torch.load(modelName))
test_model(load_model, criterion)
