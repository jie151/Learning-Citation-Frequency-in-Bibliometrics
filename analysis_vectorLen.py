import pandas as pd

def analysis_vectorLen(data_withID_filename, save_to_csv_filename):

    dataList = []
    number_of_scholars = 0

    with open(data_withID_filename, "r") as file:
        for line in file:
            word_list = line.split(" ")
            # word_list[0]: scholarID, word_list[1]: num of vectors
            dataList.append( int(word_list[1]) )

    number_of_scholars = len(dataList)
    max_vector = max(dataList)
    dataList = pd.DataFrame(dataList)
    print(dataList.describe())

    while(1):
        # input the number, determine the width of interval
        interval = (input("Enter a number: "))
        if ( not interval.isnumeric() ): break
        interval = int(interval)
        number_of_rows = int(max_vector/interval) + 1
        # create a zero-filled dataframe
        df = pd.DataFrame(0, columns=['frequency'], index=range(number_of_rows))
        print("number of rows: ", number_of_rows)
        # insert a new column "groups" recorded interval like 0-9, 10-19...
        rowName_list = []
        for i in range(number_of_rows):
            groups = str(i*interval) + "-" + str( (i+1)*interval - 1)
            rowName_list.append(groups)
        df.insert(0, "groups", rowName_list)
        # frequency distribution table
        for data in dataList[0]:
            quotient = int(data/interval) + 1
            df.at[quotient - 1, 'frequency'] = df.at[quotient - 1, 'frequency'] + 1
        print( df.head(10) )
        # probability mass function frequency / total
        pmf = df['frequency'] / number_of_scholars
        df.insert(2, "pmf", pmf)
        df.to_csv(save_to_csv_filename, index=False)

analysis_vectorLen("2022-07-28/data_withID.txt", "2022-07-28/pmf.csv")