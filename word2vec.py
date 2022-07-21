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

    max_length = 0 # the largest set of vectors
    totalSize = db.articles.estimated_document_count()
    start = 0 # control where MongoDB begins returning results
    slice_size = 1000 # the maximum number of documents/ records the cursor will return

    i = start
    current = start
    while current - start < totalSize:
        print(f"\n***********current: {current} ***********")

        if (current + slice_size) - start > totalSize: slice_size = totalSize - current

        # fetch data from mongoDB
        docs = list(db.articles.find({}).skip(current).limit(slice_size))

        collection_dataList = [] # slice_size scholars's data
        #scholarID_list = []

        for doc in docs:
            data = [] # a scholar's data
            
            # fetch scholar's profile by _id
            scholar_profile = list(db.cguscholar.find({"_id":doc['_id']}))

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

            #scholarID_list.append(doc['_id'])
            #scholarID_list.append(len(data))

            print("i: ", i, ", id: ", doc['_id'], ", len: ", len(data), " ,size: ", sys.getsizeof(data))
            i = i + 1
            max_length = len(data) if (max_length < len(data)) else max_length

            collection_dataList.append(data)

        with open(filename, 'a') as f:
            f.writelines('\n'.join([' '.join(_data) for _data in collection_dataList]))
            f.write('\n')
            #write = csv.writer(f)
            #write.writerows(collection_dataList)
        
        print(f"size: {sys.getsizeof(collection_dataList)}, max: {max_length}")

        current = current + slice_size

    print("max_length: ",max_length)

start_time = time.time()
get_scholar_profile_from_mongoDB("./data/data2.txt") #"./data/dataAll_remove_punctuation.txt"
execute = (time.time() - start_time)/60
print(f"execution time: {round(execute, 2)} mins")
