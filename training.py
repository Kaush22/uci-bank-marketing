# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:45:06 2019

@author: Kaush
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder

print("Reading Datasets")
data_train = pd.read_csv("D:\\MachineLearning\\Data_science\\uci-bank-marketing\\bank-full.csv")
data_test = pd.read_csv("D:\\MachineLearning\\Data_science\\uci-bank-marketing\\bank.csv")


#train_X = data_train.iloc[:, 0:-2].values
#train_Y = data_train.iloc[:, -1].values
#
#test_X = data_test.iloc[:,:-2].values
#test_Y = data_test.iloc[:, -1].values

train_Y = data_train[data_train.columns[-1]]
train_X = data_train.drop(columns='y')

test_Y = data_test[data_test.columns[-1]]
test_X = data_test.drop(columns='y')


labelencoder = LabelEncoder()

train_X['job'] = labelencoder.fit_transform(train_X['job'])
train_X['marital'] = labelencoder.fit_transform(train_X['marital'])
train_X['education'] = labelencoder.fit_transform(train_X['education'])
train_X['default'] = labelencoder.fit_transform(train_X['default'])
train_X['housing'] = labelencoder.fit_transform(train_X['housing'])
train_X['loan'] = labelencoder.fit_transform(train_X['loan'])
train_X['contact'] = labelencoder.fit_transform(train_X['contact'])
train_X['month'] = labelencoder.fit_transform(train_X['month'])
train_X['poutcome'] = labelencoder.fit_transform(train_X['poutcome'])

train_Y = labelencoder.fit_transform(train_Y)



test_X['job'] = labelencoder.fit_transform(test_X['job'])
test_X['marital'] = labelencoder.fit_transform(test_X['marital'])
test_X['education'] = labelencoder.fit_transform(test_X['education'])
test_X['default'] = labelencoder.fit_transform(test_X['default'])
test_X['housing'] = labelencoder.fit_transform(test_X['housing'])
test_X['loan'] = labelencoder.fit_transform(test_X['loan'])
test_X['contact'] = labelencoder.fit_transform(test_X['contact'])
test_X['month'] = labelencoder.fit_transform(test_X['month'])
test_X['poutcome'] = labelencoder.fit_transform(test_X['poutcome'])

test_Y = labelencoder.fit_transform(test_Y)



print(train_X)

cor = train_X.corr()

sns.heatmap(cor)


