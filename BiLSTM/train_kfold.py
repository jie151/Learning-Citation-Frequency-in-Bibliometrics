import numpy as np
import subprocess
import time
import torch
from   sklearn.model_selection import StratifiedKFold
from   sklearn.preprocessing import MinMaxScaler

# config
var_model_layer_num = 8         # number of recurrent layers
var_batch_size = 32
var_dataset_size = 1000         # 一次抽多少做kfold
var_input_features_num = 1200   # the number of expected features in the input x
var_hidden_features_num = 1024  # the number of features in the hidden state
var_output_dim = 1              # model's output
var_epoch_num = 10
var_kFold = 5
var_learning_rate = 0.0001
var_dropout = 0.5

# 設定訓練檔與測試檔
var_trainset_file= "../data/2022-09-21_dupli/trainset_500.txt"
var_testset_file = "../data/2022-09-21_dupli/testset_500.txt"

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
    def __init__(self, file_path, each_scholar_vectorLen):
        self.file_path = file_path
        self.each_scholar_vectorLen = each_scholar_vectorLen

    def __iter__(self):
        with open(self.file_path, "r") as file:
            for line in file:

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

def train_model(model, criterion, optimizer, scheduler, trainset_file):
    # Defind the k-fold cross validator
    kFold = StratifiedKFold(n_splits = var_kFold, random_state=202, shuffle=True)

    for epoch in range(var_epoch_num):
        # 批次從檔案中拿出一部分的資料作為dataset
        dataset = MyIterableDataset(trainset_file, var_input_features_num)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = var_dataset_size)

        for data_subset, label_subset in dataset_loader:
            # MinMaxScaler
            data_subset = data_min_max_scaler(data_subset)
            data_label_subset = MyDataset(data_subset, label_subset)

            all_fold_accuracy = {}
            all_fold_label = {}

            print(f"--------\nepoch: {epoch+1}/{var_epoch_num}, learning rate: {round(scheduler.get_last_lr()[0], 5)}")
            print(f"label/total: {count_label_ratio(label_subset)/var_dataset_size}")
            for fold, (train_indexs, valid_indexs) in enumerate(kFold.split(data_subset, label_subset)):
                print(f"train/valid size: {len(train_indexs)}/{len(valid_indexs)}")

                # Sample elements randomly from a given list of indexs, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indexs)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indexs)

                # Define data loaders for training and validating data in this fold
                trainLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = var_batch_size, sampler=train_subsampler)
                validLoader = torch.utils.data.DataLoader(data_label_subset, batch_size = var_batch_size, sampler=valid_subsampler)

                # Set loss, accuracy value
                loss_fold     = 0
                accuracy_fold = 0
                update_label  = 0

                # Iterate over the DataLoader for training data
                for inputs, labels in trainLoader:
                    # Zero the gradients
                    optimizer.zero_grad()

                    model.train()
                    model_out = model(inputs)

                    temp = (model_out >= 0.5).float()
                    accuracy_fold += (temp == labels).sum().item()
                    update_label += count_label_ratio(labels)

                    loss = criterion(model_out, labels.float())
                    loss_fold += loss
                    # 反向傳播計算每個參數的梯度值
                    loss.backward()
                    # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827
                    optimizer.step()


                # 一個fold的準確率與loss
                accuracy_fold = round(accuracy_fold*100/len(train_indexs), 3)
                loss_fold = loss_fold/len(train_indexs)

                print(f"fold {fold+1}/{var_kFold}, label(update/total): {update_label}/{len(train_indexs)}, Loss: { '{:.3f}'.format(loss_fold) }, Accuracy: {accuracy_fold}%")

                # validation theis fold
                with torch.no_grad():
                    correct = 0
                    update_label = 0
                    for valid_inputs, valid_labels in validLoader:
                        model_out = model(valid_inputs)
                        temp = (model_out >= 0.5).float()
                        correct += (temp == valid_labels).sum().item()
                        update_label += count_label_ratio(valid_labels)
                    all_fold_accuracy[fold] = 100.0 * (correct/len(valid_indexs))
                    all_fold_label[fold] = update_label/len(valid_indexs)
            print(f"********\nKFold cross validation results {var_kFold} folds")
            sum = 0.0
            for key, value in all_fold_accuracy.items():
                print(f"fold {key+1}: {round(value, 3)} %, label(update/total): {all_fold_label[key]}")
                sum += value
            print(f"average: {round(sum/len(all_fold_accuracy.items()),3)} %\n")

        #scheduler.step()


def test_model(model, criterion, testset_file):
    model.eval()
    accuracy = 0
    loss = 0
    testset = MyIterableDataset(testset_file, var_input_features_num)
    loader = torch.utils.data.DataLoader(testset, batch_size = var_batch_size)
    with torch.no_grad():
        for data, label in loader:
            # 正規化
            data = data_min_max_scaler(data)
            model_out = model(data)

            temp = (model_out >= 0.5).float()
            accuracy += (temp == label).sum().item()
            loss     += criterion(model_out, label.float())

    testset_num = int(subprocess.getstatusoutput(f"wc -l {var_testset_file}")[1].split()[0])
    loss = loss/testset_num
    accuracy = round(accuracy*100/testset_num, 3)
    print(f"testset\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")

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