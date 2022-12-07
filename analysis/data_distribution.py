# 畫學者平均論文數的CDF, PDF
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_pmf(title, xlabel, ylabel, x_data, y_data, figureName):
    ax = sns.barplot(x= x_data, y = y_data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.bar_label(ax.containers[0], labels=y_data, fmt="%.0f", padding=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figureName)

def analysis_vectorLen(citedRecord_withID_add, date):

    dataList = []
    number_of_scholars = 0

    print("read data from txt...")
    with open(citedRecord_withID_add, "r") as file:
        for index, line in enumerate(file):
            #if(index == 5):break
            word_list = line.split(" ")
            if word_list[1] != '0':
            # word_list[1] : 有幾篇論文, [2]: 有幾筆紀錄
                dataList.append( int(word_list[1]))

    #all_pmf = sns.histplot(dataList, log_scale=True, stat="density")
    number_of_scholars = len(dataList)
    max_vector = max(dataList)
    dataList = pd.DataFrame(dataList)
    print(dataList.describe())

    while(1):

        interval = (input("input the number, determine the width of interval: "))
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
            #print(data, quotient - 1, df[quotient-1][ 'frequency'])
            df.at[quotient - 1, 'frequency'] = df.at[quotient - 1, 'frequency'] + 1

        # probability mass function, cumulative distribution function
        df['pmf'] = df['frequency'] / number_of_scholars
        df['cdf'] = df['pmf'].cumsum()

        print( df.head(5) )

        # draw plot
        probability = 0.95
        pmf_size = 0
        for i in range(number_of_rows):
            if (not df.iloc[i]['cdf'] < probability):
                pmf_size = i + 1
                break
        print(pmf_size, type(pmf_size))
        title = date + "/pmf_" + str(interval) + "_" + str(probability)
        # title, xlabel, ylabel, x_data, y_data, figureName
        draw_pmf(title, "number of articles", "number of scholar (%)",df[:pmf_size]['groups'], round(df[:pmf_size]['pmf']*100, 2), title+".png")
        print(title+".png")
        # save to csv
        filename = date + "/pmf_" + str(interval) + ".csv"
        df.to_csv(filename, index=False)

#date = "../data/2022-11-20_biLSTM"
date = "."
analysis_vectorLen(date+"/citedRecord_withID_add.txt", date)