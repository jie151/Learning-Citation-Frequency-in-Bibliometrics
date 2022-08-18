import os
import time
import re
from pymongo import MongoClient
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import datetime
from langdetect import detect
from module.save_to_txt import save_to_txt
from module.remove_exist_file import remove_exist_file

# connect mongoDB
cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["CGUScholar_com"]

def get_scholarData_from_mongoDB(strData_filename, strData_withID_filename, citedRecord_withID_filename):
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
        # slice_size scholars's data/ID/citedRecord
        collection_dataList = []
        scholarID_list = []
        scholarCitedRecord_list = []

        for doc in docs:
            # a scholar's data
            data = []
            # fetch scholar's profile by _id
            scholar_profile = list(db.cguscholar.find({"_id":doc['_id']}))

            if(scholar_profile):

                record_list = []
                record_list.append(len(scholar_profile[0]['citedRecord']))
                for record in scholar_profile[0]['citedRecord']:
                    updateTime = record['updateTime'].replace("-", "").split()[0] # 2022-05-15 14:20:09 => 20220515
                    citationCount = record['cited']['citations']['All'] if record['cited'] else 0
                    record_list.extend([updateTime, citationCount])
                scholarCitedRecord_list.append(record_list)

                data.extend(scholar_profile[0]['personalData']['name'].split())
                data.extend(scholar_profile[0]['personalData']['university'].split())
                for label in scholar_profile[0]['personalData']['label']:
                    label = label.split()
                    for w in label:
                        data.extend(w.split("_"))

            not_ENarticle = 0
            for article in (doc['Articles']):
                try:
                    language = detect(article['title'])
                    if (language != "en"): not_ENarticle = not_ENarticle + 1
                except:
                    not_ENarticle = not_ENarticle + 1
                data.extend(article['publication_date'].split())
                data.extend(article['authors'].split())
                data.extend(article['source'].split())
                data.extend(article['title'].split())

            data = [re.sub("[^a-zA-Z0-9]+", "", w) for w in data]   # remove non english characters
            data = [w.lower() for w in data]                        # convert all characters to lowercase
            data = list(set(data))                                  # remove duplicates
            data = list(filter(None, data))                         # remove empty string ''

            scholarID_list.append([doc['_id'], len(doc['Articles']) - not_ENarticle])
            print(f"i: {i}, id: { doc['_id'] }, article: { len(doc['Articles']) - not_ENarticle } , len: {len(data)} ")

            i = i + 1
            max_length = len(data) if (max_length < len(data)) else max_length

            collection_dataList.append(data)

        save_to_txt(strData_filename, collection_dataList)

        scholarCitedRecord_list = [sID + record for sID, record in zip(scholarID_list, scholarCitedRecord_list)]
        save_to_txt(citedRecord_withID_filename, scholarCitedRecord_list)

        collection_dataList = [sID + collection for sID, collection in zip(scholarID_list,collection_dataList)]
        save_to_txt(strData_withID_filename, collection_dataList)

        current = current + slice_size
    print(f"max length: {max_length}")

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

def currentTime():
    now = datetime.datetime.now()
    currentTime = now.strftime("%Y-%m-%d")
    return currentTime

date  = currentTime()
path = "./" + date
strData_filename = path +"/data.txt"
strData_withID_filename = path + "/data_withID.txt"
citedRecord_withID_filename = path + "/citedRecord_withID.txt"
vector_withID_filename = path + "/vector_withID.txt"
w2v_model_filename = path + "/w2v_model.model"
w2v_vectorTable_filename = path + "/w2v_vectorTable.txt"

if not os.path.isdir(path):
    os.makedirs(path, mode = 0o777)

remove_exist_file(strData_filename)
remove_exist_file(strData_withID_filename)
remove_exist_file(vector_withID_filename)
remove_exist_file(citedRecord_withID_filename)

start_time = time.time()

get_scholarData_from_mongoDB(strData_filename, strData_withID_filename, citedRecord_withID_filename)
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