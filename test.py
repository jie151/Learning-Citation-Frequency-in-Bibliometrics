import csv
import os
import time
import sys
import string
from pymongo import MongoClient
import pandas as pd
from gensim.models import Word2Vec

# connect mongoDB
cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["CGUScholar_com"]

def get_scholar_profile_from_mongoDB(filename):
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)

    max_length = 0
    totalSize = db.articles.estimated_document_count()
    start = 0
    slice_size = 2000

    i = start
    current = start
    while current - start < totalSize:
        print("\n***********current: ",current)

        # 存放 slice_size位學者的資料
        collection_dataList = []
        scholarID_list = []

        if (current + slice_size) - start > totalSize: slice_size = totalSize - current

        # 從start開始讀取slice_size筆資料
        docs = list(db.articles.find({}).skip(current).limit(slice_size))

        for doc in docs:
            # 存放一位學者的資料
            data = []

            # 用學者ID讀取cguscholar中的profile
            scholar_profile = list(db.cguscholar.find({"_id":doc['_id']}))

            data.append(scholar_profile[0]['personalData']['name'])
            data.append(scholar_profile[0]['personalData']['university'])

            for label in scholar_profile[0]['personalData']['label']:
                label = label.split("_")
                for word in label:
                    data.append(word)

            for article in (doc['Articles']):
                data.append(article['publication_date'])

                author = article['authors'].split(", ")
                data.extend(author)

                data.append(article['source'])

                # no remove punctuation
		        #title = article['title'].split(" ")

                # remove punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
                article_title = article['title'].translate(str.maketrans('', '', string.punctuation))
                title = article_title.split(" ")
                #article_title = article['title'].replace(",", "").replace(":", "").replace(".", "") # . &  ...

                data.extend(title)

            data = list(set(data)) # remove duplicates
            data = list(filter(None, data)) # remove empty string ''

            #scholarID_list.append(doc['_id'])
            #scholarID_list.append(len(data))
            data.insert(0, doc['_id']) # insert scholar_id
            #data.insert(1,len(data))

            print("i: ", i, ", id: ", doc['_id'], ", len: ", len(data), " ,size: ", sys.getsizeof(data))
            i = i + 1
            max_length = len(data) if (max_length < len(data)) else max_length

            collection_dataList.append(data)

        #model = Word2Vec(collection_dataList, min_count=1, vector_size=1)
        #model.wv.save_word2vec_format("vector1.txt", binary=False)

        #collection_dataList = pd.DataFrame(collection_dataList)
        #collection_dataList.insert(0, "ID", scholarID_list, True)
        #collection_dataList.to_csv(filename, encoding='utf-8', index=False)
        #collection_dataList = collection_dataList.values
        #print(collection_dataList.head)
        #write slice_size data to file
        with open(filename, 'a') as f:
            write = csv.writer(f)
            write.writerows(collection_dataList)


        print(f"size: {sys.getsizeof(collection_dataList)}, max: {max_length}")

        current = current + slice_size

    print("max_length: ",max_length)
    #collection_dataList = pd.DataFrame(collection_dataList)
    #collection_dataList.insert(0, "ID", scholarID_list, True)
    #collection_dataList.to_csv(filename, encoding='utf-8', index=False)
    #collection_dataList = collection_dataList.values
    #print(collection_dataList.head)
    # write slice_size data to file
    #with open(filename, 'a') as f:
    #    write = csv.writer(f)
    #    write.writerows(collection_dataList)

start_time = time.time()
get_scholar_profile_from_mongoDB("./data/data_2.txt")
execute = (time.time() - start_time)/60
print(f"execution time: {round(execute, 2)} mins")