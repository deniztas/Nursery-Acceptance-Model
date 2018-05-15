import numpy as np
#import pandas as pd
import csv
import random
from sklearn.cross_validation import train_test_split

#from IPython import embed

def dataReader(fileName):
    ifile = open(fileName, "r")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0	
    itemList = []
    itemDict= {}

    for row in reader:
        itemList.append (row)
        itemDict[rownum] = row
        rownum += 1
        
    #print(itemList[0])
    print("\t")

    ifile.close()
    return itemList
   
#dataReader('nursery.csv')


def getDataset(data):
    data = np.array(data)
    random.shuffle(data)

    train_data = data[:70]
    test_data = data[30:]

    train_data ,test_data = train_test_split(data, test_size=0.3)

    print(train_data)
    print(len(train_data))
    print(len(test_data))
    print(type(data))

    

    return np.asarray(train_data), np.asarray(test_data)

getDataset(data = dataReader("nursery.csv"))