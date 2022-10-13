import random
import subprocess
from save_to_txt import save_to_txt
from remove_exist_file import remove_exist_file

# 把有更新、沒更新的資料分別放在不同檔案裡
def balance_apart(path):
    update_list = []
    noUpdate_list = []
    with open(path, "r") as file:
        for index, line in enumerate(file):

            data = line.split()

            if index % 5000 == 0 and index != 0:
                random.shuffle(update_list)
                random.shuffle(noUpdate_list)
                save_to_txt("update.txt", update_list)
                save_to_txt("noUpdate.txt", noUpdate_list)
                update_list = []
                noUpdate_list = []
            if (data[0] == '1'):
                update_list.append(data)
            else:
                noUpdate_list.append(data)
        save_to_txt("update.txt", update_list)
        save_to_txt("noUpdate.txt", noUpdate_list)

# 隨機挑
def balance_random (update_path, noupdate_path):
    updatefile_size = int(subprocess.getstatusoutput(f"wc -l {update_path}")[1].split()[0])
    noupdatefile_size = int(subprocess.getstatusoutput(f"wc -l {noupdate_path}")[1].split()[0])
    max_index = updatefile_size if updatefile_size < noupdatefile_size else noupdatefile_size

    print("file size: ", max_index)
    # 要拿幾筆data
    num = int(input("input a integer (data size): "))
    while(num > max_index): num = int(input("input a integer (data size): "))
    # 隨機list
    random_list = random.sample(range(0, max_index), num)
    tempFilename = f"trainset_{num*2}.txt"

    with open(update_path, "r") as file, open(noupdate_path, "r") as no_file:
        selected_list = []
        for index, (update_line, noupdate_line) in enumerate(zip(file, no_file)):

            update_line = update_line.split()
            noupdate_line=noupdate_line.split()

            if (index % 2500 == 0 and index != 0):
                random.shuffle(selected_list)
                save_to_txt(tempFilename, selected_list)
                selected_list = []
            if index in random_list:
                selected_list.append(update_line)
                selected_list.append(noupdate_line)
        save_to_txt(tempFilename, selected_list)

    with open(tempFilename, "r") as tmpFile:
        train_list = []
        test_list = []
        update_num = 0
        no_update_num = 0
        trainFile = f"../../data/2022-09-21_dupli/trainset_{num}.txt"
        testFile = f"../../data/2022-09-21_dupli/testset_{num}.txt"
        remove_exist_file(trainFile)
        remove_exist_file(testFile)
        for index, line in enumerate(tmpFile):
            line = line.split()

            if (index % 2500 == 0 and index != 0):
                save_to_txt(trainFile, train_list)
                save_to_txt(testFile, test_list)
                train_list = []
                test_list = []

            if (line[0] == '0' and no_update_num < num/2) or (line[0] == '1' and update_num < num/2):
                if line[0] == '1': update_num += 1
                else: no_update_num += 1

                train_list.append(line)
            else :
                test_list.append(line)
        save_to_txt(trainFile, train_list)
        save_to_txt(testFile, test_list)

    remove_exist_file(tempFilename)

#balance_apart("../../data/2022-09-21_dupli/dataRecord_vector_add.txt")
balance_random("./update.txt", "./noUpdate.txt")
