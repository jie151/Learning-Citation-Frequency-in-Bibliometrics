from pymongo import MongoClient

# connect mongoDB
cluster = MongoClient("mongodb://localhost:27017/")
db = cluster["CGUScholar_com"]

scholar_predict_result_file = "../data/scholar_predict_result.txt"
testset_file = "../data/testset_230000.txt"
create_file = "../data/scholarData_predict_result.txt"

# for dictionary
def save_to_txt(filename, dataList):
    with open(filename, 'a', encoding = 'utf-8') as f:
        for data in dataList:
            for _data in dataList[data] :
                f.write("%s " % _data)
            f.write("\n")

def get_scholarData_articles_from_mongoDB(scholar_predict_result_file, testset_file, create_file):

    dataList = {}
    with open(scholar_predict_result_file, 'r') as predict_file, open(testset_file, 'r') as scholar_file:
        for index, (scholar_data, predict_result) in enumerate(zip(scholar_file, predict_file)):

            scholar_data = scholar_data.split()
            predict_result = predict_result.split()

            scholarID = scholar_data[1]
            index_from_predict = int(predict_result[0])
            tempList = []
            if (index == index_from_predict) and (scholar_data[0] == predict_result[1]) :
                article_mongoDB = list(db.articles.find({"_id": scholarID}))

                if article_mongoDB:
                    print(index, scholarID, len(article_mongoDB[0]['Articles']))

                    if scholarID not in dataList:
                        tempList.append(len(article_mongoDB[0]['Articles']))
                        for article in article_mongoDB[0]['Articles']:
                            tempList.append(article['publication_date'])
                        dataList[scholarID] = tempList

                dataList[scholarID].extend([scholar_data[3], predict_result[2]]) # scholar_data[3]: 紀錄時間, predict_result[2] : 預測是否正確
    save_to_txt(create_file, dataList)

get_scholarData_articles_from_mongoDB(scholar_predict_result_file, testset_file, create_file)