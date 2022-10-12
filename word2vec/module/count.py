#計算有更新、沒更新的比例
def count_isChange(path):
    with open(path, "r") as file:
        update = 0
        no_update = 0
        for index, line in enumerate(file):

            data = line.split()

            if (data[0] == "1"):
                update += 1
            else :
                no_update +=1
        sum = update + no_update
        print(f"update:    {update}, {round(update/sum,3)} ")
        print(f"no update: {no_update}, {round(no_update/sum,3)} ")

count_isChange("../data/2022-08-26/dataRecord_word_add.txt")
