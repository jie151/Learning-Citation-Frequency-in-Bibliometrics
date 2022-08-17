import pandas as pd
from module.save_to_txt import save_to_txt

# 會去掉 article數 < 1 且 record < 輸入的n的學者
def generate_data_to_txt(vectorWithID_filename, citedRecordWithID_filename):
    # citedRecord column: ID, number of articles, number of records , time0, citation0, time1, citation1 ...
    recordLen_dataframe = pd.read_csv(citedRecordWithID_filename, sep=" ", header = None, usecols= [2])
    print(f"citedRecord:\n{round(recordLen_dataframe[2].describe(), 2)}")
    max = recordLen_dataframe[2].max()

    n = int(input("enter an integer, determine which record: "))
    while(n > max):
        n = int(input("the number is too big, input again: "))

    current_updateTime_index = 3 + 2 * n

    with open(vectorWithID_filename, "r") as vectorFile, open(citedRecordWithID_filename, "r") as recordFile :

        all_record_vectorList = []

        for index, (each_vector, each_record) in enumerate(zip(vectorFile, recordFile)):
            if index % 5000 == 0 and index > 0:
                save_to_txt(date + "/dataRecord_"+ str(n)+".txt", all_record_vectorList)
                all_record_vectorList = []
            vector = each_vector.split()
            record = each_record.split()
            # check ID
            if (vector[0] == record[0]):
                # check number of record > n and number of
                if ( int(record[2]) > n and int(record[1]) > 1):
                    is_change = 1 if (n > 0 and record[current_updateTime_index + 1] != record[current_updateTime_index - 1 ]) or n == 0 else 0
                    record_vector = [record[current_updateTime_index],record[current_updateTime_index + 1], is_change]
                    record_vector.extend(vector[1:])
                    all_record_vectorList.append(record_vector)
        if len(all_record_vectorList) > 0: save_to_txt(date + "/dataRecord_"+ str(n)+".txt", all_record_vectorList)

date = "./2022-08-15"
generate_data_to_txt( date + "/vector_withID.txt", date + "/citedRecord_withID.txt")
