from pymongo import MongoClient
import pandas as pd
from gensim.models import Word2Vec

# connect mongoDB


db = cluster["CGUScholar"]

def get_scholar_profile_from_mongoDB(scholarID_list):
    collection_dataList = []
    docs = list(db.cguscholar.find({}))

    i = 0
    for doc in docs:
        if i > 10: break
        data = []
        scholarID_list.append(doc['_id'])
        data.append(doc['personalData']['name'])
        data.append(doc['personalData']['university'])

        for label in (doc['personalData']['label']):
            label = label.split("_")

            for word in label:
                data.append(word)
        collection_dataList.append(data)

        i = i + 1
        #print("i:", i)
    return collection_dataList

def get_vector(rowData_list):
    all_vector = []
    for scholar in rowData_list:
        vector = []
        for word in scholar:
            vector.append(model.wv.get_vector(word)[0])
        all_vector.append(vector)
    return all_vector

scholarID_list = []
rowData_list = get_scholar_profile_from_mongoDB(scholarID_list)
print(rowData_list)

rowData = pd.DataFrame(rowData_list)
rowData.insert(0, "ID", scholarID_list, True)
print(rowData)

model = Word2Vec(rowData_list, min_count = 1, vector_size = 1)  # Word2Vec, build_vocab, train
model.save("test.model")
model.wv.save_word2vec_format('test.txt', binary=False)

# get word vector
vector = get_vector(rowData_list)
print(vector)