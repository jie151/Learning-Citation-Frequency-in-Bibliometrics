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

    # remove a file if exists
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)
    if (os.path.exists("./data/scholarID.csv") and os.path.isfile("./data/scholarID.csv")):
        os.remove("./data/scholarID.csv")

    max_length = 0 # the largest set of vectors
    totalSize = 5000#db.articles.estimated_document_count()
    start = 0 # control where MongoDB begins returning results
    slice_size = 5000 # the maximum number of documents/ records the cursor will return

    i = start
    current = start
    while current - start < totalSize:
        print(f"\n***********current: {current} ***********")

        if (current + slice_size) - start > totalSize: slice_size = totalSize - current

        # fetch data from mongoDB
        docs = list(db.articles.find({}).skip(current).limit(slice_size))
        # slice_size scholars's data
        collection_dataList = []
        scholarID_list = []

        for doc in docs:
            # a scholar's data
            data = []
            # fetch scholar's profile by _id
            scholar_profile = list(db.cguscholar.find({"_id":doc['_id']}))

            if(scholar_profile):
                data.extend(scholar_profile[0]['personalData']['name'].split(" "))
                data.extend(scholar_profile[0]['personalData']['university'].split(" "))

                for label in scholar_profile[0]['personalData']['label']:
                    label = label.split("_")
                    for word in label: data.append(word)

            for article in (doc['Articles']):
                data.append(article['publication_date'])

                author = article['authors'].split(" ")
                data.extend(author)

                article_source = article['source'].translate(str.maketrans('', '', string.punctuation))
                data.extend(article_source.split(" "))

                # remove punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
                article_title = article['title'].translate(str.maketrans('', '', string.punctuation))
                data.extend(article_title.split(" "))
                #article_title = article['title'].replace(",", "").replace(":", "").replace(".", "") # . &  ...

            data = list(set(data)) # remove duplicates
            data = list(filter(None, data)) # remove empty string ''
            data.insert(0, doc['_id']) # insert scholar_id
            #data.insert(1,len(data))

            scholarID_list.append(doc['_id'])
            scholarID_list.append(len(data))

            print("i: ", i, ", id: ", doc['_id'], ", len: ", len(data), " ,size: ", sys.getsizeof(data))
            i = i + 1
            max_length = len(data) if (max_length < len(data)) else max_length

            collection_dataList.append(data)

        with open(filename, 'a') as f:
            f.writelines('\n'.join([' '.join(_data) for _data in collection_dataList]))
            f.write('\n')
        with open("./data/scholarID.csv", 'a') as f:
            write = csv.writer(f)
            write.writerows(scholarID_list)

        print(f"size: {sys.getsizeof(collection_dataList)}, max: {max_length}")

        current = current + slice_size

    print("max_length: ",max_length)

start_time = time.time()
get_scholar_profile_from_mongoDB("./data/data.txt") #"./data/dataAll_remove_punctuation.txt"
execute = (time.time() - start_time)
print("execute time : ",time.strftime("%H:%M:%S", time.gmtime(execute)))
