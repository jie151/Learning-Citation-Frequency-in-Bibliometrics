import os
import time
import sys
import re
from pymongo import MongoClient
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import datetime

# connect mongoDB
cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["CGUScholar_com"]

def save_to_txt(filename, dataList):
    with open(filename, 'a', encoding = 'utf-8') as f:
        for data in dataList:
            for _data in data :
                f.write("%s " % _data)
            f.write("\n")

def get_scholar_profile_from_mongoDB(strData_filename, strData_withID_filename):
    max_length = 0 # the largest set of vectors
    totalSize = db.articles.estimated_document_count()

    start = 0 # control where MongoDB begins returning results
    slice_size = 5000 # the maximum number of documents/ records the cursor will return

    i = start
    current = start
    while current - start < totalSize:

        if (current + slice_size) - start > totalSize: slice_size = totalSize - (current - start)
        print(f"\n***********current: {current} ***********, slice_size: {slice_size}")

        # fetch data from mongoDB
        docs = list(db.articles.find({}).skip(current).limit(slice_size))
        # slice_size scholars's data/ID
        collection_dataList = []
        scholarID_list = []

        for doc in docs:
            # a scholar's data
            data = []
            # fetch scholar's profile by _id
            scholar_profile = list(db.cguscholar.find({"_id":doc['_id']}))

            if(scholar_profile):
                data.extend(scholar_profile[0]['personalData']['name'].split())
                data.extend(scholar_profile[0]['personalData']['university'].split())
                for label in scholar_profile[0]['personalData']['label']:
                    label = label.split()
                    for w in label:
                        data.extend(w.split("_"))

            for article in (doc['Articles']):
                data.extend(article['publication_date'].split())
                data.extend(article['authors'].split())
                data.extend(article['source'].split())
                data.extend(article['title'].split())

            # remove non english characters
            data = [re.sub("[^a-zA-Z0-9]+", "", w) for w in data]
            # convert all characters to lowercase
            data = [w.lower() for w in data]
            # remove duplicates
            data = list(set(data))
            # remove empty string ''
            data = list(filter(None, data))

            scholarID_list.append([doc['_id'], len(data)])

            print("i: ", i, ", id: ", doc['_id'], ", len: ", len(data), " ,size: ", sys.getsizeof(data))
            i = i + 1
            max_length = len(data) if (max_length < len(data)) else max_length

            collection_dataList.append(data)

        save_to_txt(strData_filename, collection_dataList)
        collection_dataList = [ID + collection for ID, collection in zip(scholarID_list,collection_dataList)]

        save_to_txt(strData_withID_filename, collection_dataList)

        current = current + slice_size
    print(f"size: {sys.getsizeof(collection_dataList)}, max: {max_length}")

def train_W2V_model(strData_filename, w2v_model_filename, w2v_vectorTable_filename):
    model = Word2Vec(LineSentence(strData_filename),  min_count = 1, vector_size = 1)
    model.save(w2v_model_filename)
    model.wv.save_word2vec_format(w2v_vectorTable_filename, binary=False)
    return model

def get_vector(model, strData_list):
    vector = []
    for word in strData_list:
        vector.append(model.wv.get_vector(word)[0])
    return vector

def word_embedding(strData_withID_filename,vector_withID_filename, model):

    with open(strData_withID_filename, encoding="utf-8") as f:
        all_vectorList = []

        for index, line in enumerate(f):
            if index % 5000 == 0 and index > 0:
                save_to_txt(vector_withID_filename, all_vectorList)
                print("open file, index: ", index)
                all_vectorList = []

            line_list = line.split()
            if line_list[-1] == "\n": line_list.pop()

            vector_list = get_vector(model, line_list[2:])
            vector_list.insert(0, line_list[0])

            all_vectorList.append(vector_list)

        save_to_txt(vector_withID_filename, all_vectorList)

def remove_exist_file(filename):
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)

def currentTime():
    now = datetime.datetime.now()
    currentTime = now.strftime("%Y-%m-%d")
    return currentTime

date  = currentTime()
path = "./" + date
strData_filename = path +"/data.txt"
strData_withID_filename = path + "/data_withID.txt"
vector_withID_filename = path + "/vector_withID.txt"
w2v_model_filename = path + "/w2v_model.model"
w2v_vectorTable_filename = path + "/w2v_vectorTable.txt"

if not os.path.isdir(path):
    os.makedirs(path, mode = 0o777)

remove_exist_file(strData_filename)
remove_exist_file(strData_withID_filename)
remove_exist_file(vector_withID_filename)

start_time = time.time()
get_scholar_profile_from_mongoDB(strData_filename, strData_withID_filename)
execute = (time.time() - start_time)
print("fetch data from DB : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

start_time = time.time()
model = train_W2V_model(strData_filename, w2v_model_filename, w2v_vectorTable_filename)
#model = Word2Vec.load(w2v_model_filename)
execute = (time.time() - start_time)
print("train model : ",time.strftime("%H:%M:%S", time.gmtime(execute)))

start_time = time.time()
word_embedding(strData_withID_filename,vector_withID_filename, model)
execute = (time.time() - start_time)
print("word embedding : ",time.strftime("%H:%M:%S", time.gmtime(execute)))