import random
import subprocess
from module.save_to_txt import save_to_txt
from module.remove_exist_file import remove_exist_file

# 把有更新、沒更新的資料分別放在不同檔案裡
def apart_update_noUpdate(path, update_file, noUpdate_file):
    update_list   = []
    noUpdate_list = []

    remove_exist_file(update_file)
    remove_exist_file(noUpdate_file)
    with open(path, "r") as file:
        for index, line in enumerate(file):
            data = line.split()

            if index % 5000 == 0 and index != 0:
                random.shuffle(update_list)
                random.shuffle(noUpdate_list)
                save_to_txt(update_file, update_list)
                save_to_txt(noUpdate_file, noUpdate_list)
                update_list   = []
                noUpdate_list = []

            if (data[0] == '1'):
                update_list.append(data)
            else:
                noUpdate_list.append(data)
        random.shuffle(update_list)
        random.shuffle(noUpdate_list)
        save_to_txt(update_file, update_list)
        save_to_txt(noUpdate_file, noUpdate_list)

def balance_random(update_path, noUpdate_path, dataset_path):
    updatefile_size = int(subprocess.getstatusoutput(f"wc -l {update_path}")[1].split()[0])
    noUpdatefile_size = int(subprocess.getstatusoutput(f"wc -l {noUpdate_path}")[1].split()[0])
    max_index = updatefile_size if updatefile_size < noUpdatefile_size else noUpdatefile_size
    print("***file size: ", max_index)

    # 要拿幾筆資料
    num = int(input("input a integer (data size): "))
    while(num > max_index): num = int(input(f"input again, the number should < {max_index}: "))
    # 隨機list
    random_list = random.sample(range(0, max_index), num)
    temp_filename = f"trainset_{num*2}.txt"

    with open(update_path, "r") as file, open(noUpdate_path, "r") as no_file:
        selected_list = []
        for index, (update_line, noUpdate_line) in enumerate(zip(file, no_file)):
            update_line   = update_line.split()
            noUpdate_line = noUpdate_line.split()

            if index % 2500 == 0 and index != 0:
                random.shuffle(selected_list)
                save_to_txt(temp_filename, selected_list)
                selected_list = []
            if index in random_list:
                selected_list.append(update_line)
                selected_list.append(noUpdate_line)
        random.shuffle(selected_list)
        save_to_txt(temp_filename, selected_list)

    with open(temp_filename, "r") as tmpFile:
        train_list = []
        test_list  = []
        update_num   = 0
        noUpdate_num = 0
        trainFile = f"{dataset_path}/trainset_{num}.txt"
        testFile  = f"{dataset_path}/testset_{num}.txt"
        remove_exist_file(trainFile)
        remove_exist_file(testFile)
        for index, line in enumerate(tmpFile):
            line = line.split()

            if (index % 2500 == 0 and index != 0):
                random.shuffle(train_list)
                random.shuffle(test_list)
                save_to_txt(trainFile, train_list)
                save_to_txt(testFile, test_list)
                train_list = []
                test_list  = []

            if (line[0] == '0' and noUpdate_num < num/2) or (line[0] == '1' and update_num < num/2):
                if line[0] == '1': update_num += 1
                else: noUpdate_num += 1
                train_list.append(line)
            else :
                test_list.append(line)
        random.shuffle(train_list)
        random.shuffle(test_list)
        save_to_txt(trainFile, train_list)
        save_to_txt(testFile, test_list)
        print(f"create file {trainFile}, {testFile}")
    remove_exist_file(temp_filename)

#path = "../../data/2022-09-21_dupli"
#update_file = path + "/update.txt"
#noUpdate_file = path + "/noUpdate.txt"

#apart_update_noUpdate(path + "/dataRecord_word_add.txt", update_file, noUpdate_file)
#balance_random(update_file, noUpdate_file, path)