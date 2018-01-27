# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:36:20 2018

@author: ASSG
"""

import os
import sys

def append_to_path(relative_path):
    abs_path = os.path.abspath(relative_path)
    print(abs_path)
    if abs_path not in sys.path:
        sys.path.append(abs_path)

append_to_path("/src")
#append_to_path("ci")

#%% import general libraries
import numpy as np
import pandas as pd

#%% import preprocessor classes developed for the House Price prediction kaggle project
from house_price_prediction.framework_extensions.HousePricePred_mean_mode_preprocessor import Mean_Mode_Preprocessor
from house_price_prediction.framework_extensions.HousePricePred_mean_price_label_encoder import Mean_Price_Preprocessor

#%% local function that receives a kind of preprocessor and applies the assertions common to each one
# (can't have "test_" in the name otherwise pytest will call it)
def localtester_hpp_preprocessor(preprocessor):
    data_preprocessor = preprocessor(["SalePrice"])
    train_dataframe = pd.read_csv('house_price_prediction/input/train.csv', index_col='Id')
    test_dataframe = pd.read_csv('house_price_prediction/input/test.csv', index_col='Id')

    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)

    #tests
    #test if there are any NaNs
    assert(X_train.isnull().sum().sum() == 0)
    assert(X_test.isnull().sum().sum() == 0)

    #test the outputs have the correct shapes
    assert(X_train.shape == (1460, 79))
    assert(y_train.shape == (1460,))
    assert(X_test.shape == (1459, 79))

    # test every column in the outputs is of numeric type
    assert(X_train.select_dtypes(exclude = [np.number]).shape[1] == 0)
    assert(X_test.select_dtypes(exclude = [np.number]).shape[1] == 0)

#%% function called automaitcally by pytest to execute the tests
def test_hpp():
	localtester_hpp_preprocessor(Mean_Mode_Preprocessor)
	localtester_hpp_preprocessor(Mean_Price_Preprocessor)
