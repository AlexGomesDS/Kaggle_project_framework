# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre
"""

#%% 0.1 Import libraries
import numpy as np
import pandas as pd

from features.preprocessor.mean_mode_preprocessor import Mean_Mode_Preprocessor

#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

#%% 1.0 Preprocess the data
data_preprocessor = Mean_Mode_Preprocessor()
# datasets after preprocessing and feature engineering
eng_train_dataset = data_preprocessor.prepare(train_dataframe)
eng_test_dataset = data_preprocessor.cook(test_dataframe)


#%% 