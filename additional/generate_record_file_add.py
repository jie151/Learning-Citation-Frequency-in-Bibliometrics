# 生成每個學者的紀錄檔，包括h_index, i10_index
from tabnanny import filename_only
from pymongo import MongoClient
from module.save_to_txt import save_to_txt
from module.remove_exist_file import remove_exist_file

# connect mongoDB
cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["CGUScholar_com"]

def generate_record_file(read_data_withID_file, filename):
    recordList = []

    with open(read_data_withID_file, "r") as file:
        for index, data in enumerate(file):
            if (index % 8000 == 0 and index > 0):
                print(index)
                save_to_txt(filename, recordList)
                recordList = []
            data = data.split(" ")[:2]
            scholar_profile = list(db.cguscholar.find({"_id":data[0]}))

            if(scholar_profile):
                data.append(len(scholar_profile[0]['citedRecord']))
                for record in scholar_profile[0]['citedRecord']:
                    updateTime = record['updateTime'].replace("-", "").split()[0] # 2022-05-15 14:20:09 => 20220515
                    if record['cited']:
                        citations_all = record['cited']['citations']['All']
                        h_index_all = record['cited']['h_index']['All']
                        i10_index_all = record['cited']['i10_index']['All']
                    else:
                        citations_all = 0
                        h_index_all   = 0
                        i10_index_all = 0

                    data.extend([updateTime, citations_all,h_index_all,i10_index_all])
            recordList.append(data)
        if(len(recordList) > 0): save_to_txt(filename, recordList)

date = "./2022-08-26"
read_data_withID_file = date + "/data_withID.txt"
filename = date + "/citedRecord_withID_add.txt"

remove_exist_file(filename)

print("create file...")
generate_record_file(read_data_withID_file, filename)
print("create ", filename_only)