import random
import subprocess
from save_to_txt import save_to_txt

# 把有更新、沒更新的資料分別放在不同檔案裡
def balance_apart(path):
    update_list = []
    noUpdate_list = []
    with open(path, "r") as file:
        for index, line in enumerate(file):

            data = line.split()

            if index % 5000 == 0 and index != 0:
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
def balance_random (path):
    max_index = int(subprocess.getstatusoutput(f"wc -l {path}")[1].split()[0])
    print("file size: ", max_index)
    # 要拿幾筆data
    num = int(input("input a integer (data size): "))
    while(num > max_index): num = int(input("input a integer (data size): "))
    # 隨機list
    random_list = random.sample(range(0, max_index), num)


    with open(path, "r") as file:
        selected_list = []
        for index, line in enumerate(file):
            line = line.split()
            if (index % 5000 == 0 and index != 0):
                save_to_txt(f"balance_{num}.txt", selected_list)
                selected_list = []
            if index in random_list:
                selected_list.append(line)
        save_to_txt(f"balance_{num}.txt", selected_list)

#balance_apart("../2022-08-26/testset_dataRecord_vector_add.txt")

balance_random("./noUpdate.txt")
balance_random("./update.txt")