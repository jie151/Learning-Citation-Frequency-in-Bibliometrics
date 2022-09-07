#有含label資料，看下次是否真的有更新，如n = 2, 有存n = 3 - n = 2時的引用次數，假設學者有五筆紀錄，只會產生1, 2, 3的 data，因為最新的那筆沒有label，同時第4筆當測試集
import subprocess
from module.save_to_txt import save_to_txt
from module.remove_exist_file import remove_exist_file

def generate_a_data(record_list, n_record):
    current_updateTime_index = 4 * n_record - 1
    if (n_record > 1):
        curr_isChange = 1 if (record_list[current_updateTime_index + 1] != record_list[current_updateTime_index - 1 ]) else 0
    else:
        curr_isChange = 1
    # 下一次是否有增加，如_record = 2, 確認第3筆與第2筆時的引用次數是否不同(做label)
    next_citation = int(record_list[4*n_record + 4])
    current_citation = int(record_list[4*n_record])
    next_isChange = 1 if (next_citation - current_citation != 0) else 0

    record_vector = [ next_isChange, record_list[0], curr_isChange]
    record_vector.extend(record_list[current_updateTime_index:current_updateTime_index + 4]) # update time + citation + h_index + i10_index
    return record_vector

# 會去掉 article數 < 1 且 record < 輸入的n的學者
def generate_data_to_txt(word_or_vector, read_word_or_vector_file, citedRecordWithID_file, trainsetFile, testsetFile):
    print("create file ...")
    with open( read_word_or_vector_file, "r") as vectorFile, open(citedRecordWithID_file, "r") as recordFile :
        all_record_vectorList = []
        all_testsetList = []
        for index, (each_vector, each_record) in enumerate(zip(vectorFile, recordFile)):
            if index % 3000 == 0 and index > 0: # 每3000名學者就存起來
                save_to_txt(trainsetFile, all_record_vectorList)
                save_to_txt(testsetFile, all_testsetList)
                all_record_vectorList = []
            vector = each_vector.split()
            record = each_record.split()
            # 判斷ID是否相同與record的長度是否大於2，可能有article有，但cguscholar沒有的情況
            if (vector[0] == record[0] and len(record) > 2):
                num_record = int(record[2])
                # check number of articles > 1
                if (int(record[1]) > 1):
                    for n_record in range(1, num_record - 1):
                        record_vector = generate_a_data(record, n_record)
                        record_vector.extend(vector[word_or_vector:]) # vector file: ID + vectors , word file : ID + num of articles + words
                        all_record_vectorList.append(record_vector)
                    if (num_record - 1 > 1):
                        # 把倒數第二筆資料當測試集的資料
                        testsetList = generate_a_data(record, num_record - 1)
                        testsetList.extend(vector[word_or_vector:])
                        all_testsetList.append(testsetList)
            else:
                if (len(record) > 2):
                    print(f"warning! {index}, {vector[0]} : {record[0]}, length: {len(record)}")

        if len(all_record_vectorList) > 0:
            save_to_txt(trainsetFile, all_record_vectorList)
            save_to_txt(testsetFile, all_testsetList)
        train_cnt = subprocess.getstatusoutput(f"wc -l {trainsetFile}")[1].split()[0]
        test_cnt = subprocess.getstatusoutput(f"wc -l {testsetFile}")[1].split()[0]
        print(f"trainset: {train_cnt}, testset: {test_cnt}")
        print(f"{ trainsetFile } created")
        print(f"{ testsetFile } created")
date = "./2022-08-15"

word_or_vector = int(input("input 1 or 2 (1: vector for biLSTM, 2: word for bert): "))

if (word_or_vector == 1):
    trainsetFile = date + "/trainset_dataRecord_vector_add.txt"
    testsetFile = date + "/testset_dataRecord_vector_add.txt"
    read_word_or_vector_file = date + "/vector_withID.txt"
else:
    trainsetFile = date + "/trainset_dataRecord_word_add.txt"
    testsetFile = date + "/testset_dataRecord_word_add.txt"
    read_word_or_vector_file = date + "/data_withID.txt"

remove_exist_file(trainsetFile)
remove_exist_file(testsetFile)
generate_data_to_txt(word_or_vector, read_word_or_vector_file, date + "/citedRecord_withID_add.txt", trainsetFile, testsetFile)