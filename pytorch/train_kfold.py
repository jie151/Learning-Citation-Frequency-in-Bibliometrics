import numpy as np
import  subprocess
import  time
import  torch
from sklearn.model_selection import StratifiedKFold
from module.save_to_txt import save_to_txt

# config
batch_size = 16
input_features = 1300   # the number of expected features in the input x
hidden_feature_dim = 128 # the number of features in the hidden state
model_layer_num = 3     # number of recurrent layers
output_dim = 1          # model's output
epochs_num = 10
kFolds_num = 5
# 設定訓練檔與測試檔
trainFile= "./dataRecord_vector_2_397.txt"
testFile = "./test_t.txt"
#trainFile= "../../../word_embedding_env/w2v/2022-08-26/trainset_dataRecord_vector_add.txt"
#testFile = "../../../word_embedding_env/w2v/2022-08-26/testset_dataRecord_vector_add.txt"
trainset_num = subprocess.getstatusoutput(f"wc -l {trainFile}")[1].split()[0]
testset_num = subprocess.getstatusoutput(f"wc -l {testFile}")[1].split()[0]
print(f"train: {trainset_num}, test: {testset_num} ")
train_arg_list =  {"path": trainFile, "each_scholar_vectorLen": input_features}
test_arg_list = {"path": testFile, "each_scholar_vectorLen": input_features}

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
                #print(f"{data[1]} train.shape: {train.shape}, label.shape: {label.shape} ")
                train = torch.tensor(train)
                label = torch.tensor(label)
                yield(train, label)

class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_features, hidden_size =  hidden_feature_dim, num_layers = model_layer_num,
                            bidirectional=True) # input, output: ( seq, batch_size, feature)
        # fully connected layer
        self.fc =torch.nn.Linear(hidden_feature_dim*2, output_dim)

    def forward(self, X):
        # X: [batch_size, seq, feature]
        batch_size = X.shape[0]

        input = X.transpose(0, 1)  # input : [seq, batch_size, feature]

        hidden_state=torch.randn(model_layer_num * 2, batch_size, hidden_feature_dim) # [num_layers * num_directions, batch, hidden_size]
        cell_state  =torch.randn(model_layer_num * 2, batch_size, hidden_feature_dim) # [num_layers * num_directions, batch, hidden_size]

        model_out, (h_n, c_n) = self.lstm(input, (hidden_state, cell_state))
        #print(model_out.shape) (1, 10:batch, 100:feature)
        output = self.fc(model_out[-1,:,:])
        return output

def train_model(model, kFolds_num, epochs_num, batch_size):
    # mean squared error : input x and target y.
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Defind the k-fold cross validator
    kFold = StratifiedKFold(n_splits = kFolds_num, random_state=202, shuffle=True)

    for epoch in range(epochs_num):
        # 批次從檔案中拿出一部分的資料作為dataset
        dataset = MyIterableDataset(train_arg_list)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 3000, shuffle=False)

        for data_subset, label_subset in loader:
            all_fold_accuracy = {}
            for fold, (train_indexs, valid_indexs) in enumerate(kFold.split(data_subset, label_subset)):

                print(f"--------\nepoch: {epoch+1}/{epochs_num}, train/valid size: {len(train_indexs)}/{len(valid_indexs)}")

                train_label_fold = label_subset[train_indexs]
                valid_label_fold = label_subset[valid_indexs]

                # Sample elements randomly from a given list of indexs, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indexs)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indexs)

                # Define data loaders for training and validating data in this fold
                trainLoader = torch.utils.data.DataLoader(data_subset, batch_size = batch_size, sampler=train_subsampler, shuffle=False)
                validLoader = torch.utils.data.DataLoader(data_subset, batch_size = batch_size, sampler=valid_subsampler, shuffle=False)

                # Set loss, accuracy value
                loss_fold     = 0
                accuracy_fold = 0
                # Iterate over the DataLoader for training data
                for inputs in trainLoader:
                    model.train()
                    # Zero the gradients
                    optimizer.zero_grad()

                    model_out = model(inputs)
                    # if model_out > 0.5, result = 1
                    result = (model_out > 0.5).float()
                    labels  = train_label_fold.to(torch.float32)
                    # 計算accuracy
                    for _label, _result in zip(labels, result):
                        if (_label == _result):
                            accuracy_fold += 1

                    # 將tensor -> list, 存預測值與label
                    #resultList = result.tolist()
                    loss = criterion(model_out, labels)
                    loss.backward() # 反向傳播計算每個參數的梯度值
                    optimizer.step() # 用梯度下降來更新參數值 # 參考:https://blog.csdn.net/PanYHHH/article/details/107361827
                    loss_fold += loss
                # 一個fold的準確率與loss
                accuracy_fold = round(accuracy_fold*100/len(train_indexs), 3)
                loss_fold = loss_fold/len(train_indexs)

                print(f"fold {fold+1}/{kFolds_num}, Loss: { '{:.3f}'.format(loss_fold) }, Accuracy: {accuracy_fold}%")

                # save model
                save_path = f"./model-fold-{fold}.pth"
                torch.save(model.state_dict(), save_path)

                # validation theis fold
                with torch.no_grad():
                    correct = 0
                    for valid_inputs in validLoader:
                        #valid_label_fold
                        model_out = model(valid_inputs)
                        result = (model_out > 0.5).float()
                        label = valid_label_fold
                        for _label, _result in zip(label, result):
                            if (_label == _result):
                                correct += 1
                    all_fold_accuracy[fold] = 100.0 * (correct/len(valid_label_fold))
            print(f"********\nKFold cross validation results {kFolds_num} folds")
            sum = 0.0
            for key, value in all_fold_accuracy.items():
                print(f"fold {key+1}: {round(value, 3)} %")
                sum += value
            print(f"average: {round(sum/len(all_fold_accuracy.items()),3)} %\n")

def test_model(model):
    model.eval()
    criterion = torch.nn.MSELoss()
    accuracy = 0
    loss = 0
    testset = MyIterableDataset(test_arg_list)
    loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False)
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
            loss += criterion(model_out, label)

    loss = loss/int(testset_num)
    accuracy = round(accuracy*100/int(testset_num), 3)
    print(f"test\nLoss: { '{:.3f}'.format(loss) }, Accuracy: {accuracy}%")


start_time = time.time()
#train(num_epochs)
model = BiLSTM()
train_model(model, kFolds_num, epochs_num, batch_size)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

modelName = "model_state_dict.pt"
torch.save(model.state_dict(), modelName)
print(f"save model's parameter: {modelName}")

# Load model
load_model = BiLSTM()
load_model.load_state_dict(torch.load(modelName))
test_model(load_model)