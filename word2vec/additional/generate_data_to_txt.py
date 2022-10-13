#有含label資料，看下次是否真的有更新，如n = 2, 有存n = 3 - n = 2時的引用次數，假設學者有五筆紀錄，只會產生1, 2, 3, 4的 data，因為最新的那筆沒有label
import pandas as pd
from module.save_to_txt import save_to_txt
from module.remove_exist_file import remove_exist_file

def get_next_record_isChange(record_list, n):
    next_citation = int(record_list[2*n + 4])
    current_citation = int(record_list[2*n + 2])
    isChange = 1 if (next_citation - current_citation != 0) else 0
    #print(f"{record_list[0]}, curr: {current_citation}, next: {next_citation}, isChange:{isChange}")
    return isChange

# 會去掉 article數 < 1 且 record < 輸入的n的學者
def generate_data_to_txt(word_or_vector, read_word_or_vector_file, citedRecordWithID_file, filename):
    # citedRecord column: ID, number of articles, number of records , time0, citation0, time1, citation1 ...
    recordLen_dataframe = pd.read_csv(citedRecordWithID_file, sep=" ", header = None, usecols= [2])
    print(f"citedRecord:\n{round(recordLen_dataframe[2].describe(), 2)}")

    remove_exist_file(filename)
    print("create file ...")

    with open( read_word_or_vector_file, "r") as vectorFile, open(citedRecordWithID_file, "r") as recordFile :

        all_record_vectorList = []
        cnt = 0
        for index, (each_vector, each_record) in enumerate(zip(vectorFile, recordFile)):
            if index % 3000 == 0 and index > 0: # 每3000名學者就存起來
                save_to_txt(filename, all_record_vectorList)
                all_record_vectorList = []
            vector = each_vector.split()
            record = each_record.split()
            # 判斷ID是否相同與record的長度是否大於2，可能有article有，但cguscholar沒有的情況
            if (vector[0] == record[0] and len(record) > 2):

                num_record = int(record[2])
                # check number of articles > 1
                if (int(record[1]) > 1):
                    for n_record in range(1, num_record):
                        cnt += 1
                        current_updateTime_index = 1 + 2 * n_record
                        # 確認這一次是否有增加，如_record = 2, 確認第2筆與第1筆時的引用次數是否不同， default:1
                        if (n_record > 1):
                            curr_isChange = 1 if (record[current_updateTime_index + 1] != record[current_updateTime_index - 1 ]) else 0
                        else:
                            curr_isChange = 1
                        # 下一次是否有增加，如_record = 2, 確認第3筆與第2筆時的引用次數是否不同(做label)
                        next_isChange = get_next_record_isChange(record, n_record)
                        record_vector = [next_isChange,record[0],record[current_updateTime_index],record[current_updateTime_index + 1], curr_isChange]
                        record_vector.extend(vector[word_or_vector:]) # vector file: ID + vectors , word file : ID + num of articles + words
                        all_record_vectorList.append(record_vector)
            else:
                if (len(record) > 2):
                    print(f"warning! {index}, {vector[0]} : {record[0]}, length: {len(record)}")


        print(f"number of data: {cnt}")
        if len(all_record_vectorList) > 0: save_to_txt(filename, all_record_vectorList)
        print(f"{ filename } created")

date = "../data/2022-08-26"

word_or_vector = int(input("input 1 or 2 (1: vector for biLSTM, 2: word for bert): "))

if (word_or_vector == 1):
    filename = date + "/dataRecord_vector.txt"
    read_word_or_vector_file = date + "/vector_withID.txt"
else:
    filename = date + "/dataRecord_word.txt"
    read_word_or_vector_file = date + "/data_withID.txt"

generate_data_to_txt(word_or_vector, read_word_or_vector_file, date + "/citedRecord_withID.txt", filename)