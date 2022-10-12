def save_to_txt(filename, dataList):
    with open(filename, 'a', encoding = 'utf-8') as f:
        for data in dataList:
            for _data in data :
                f.write("%s " % _data)
            f.write("\n")