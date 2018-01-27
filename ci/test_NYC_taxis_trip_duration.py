# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:01:26 2018

@author: Alexandre
"""

import os
import sys

def append_to_path(relative_path):
    abs_path = os.path.abspath(relative_path)
    print(abs_path)
    if abs_path not in sys.path:
        sys.path.append(abs_path)
        
append_to_path("src")
append_to_path("ci")

#%% import general libraries
import numpy as np
import pandas as pd

#%% import preprocessor classes developed for the NYC taxi trip duration kaggle project
from NYC_taxi_trip_duration.framework_extensions.NYC_taxis_preprocessor import NYCTaxisPreprocessor
from NYC_taxi_trip_duration.framework_extensions.NYC_taxis_feature_selector import NYCTaxisFeatureSelector

#%% function called automaitcally by pytest to execute the tests
def test_taxis_preprocessor():
    train_dataframe = pd.read_csv('ci/NYC_taxi_trip_duration/input/train1000.csv', index_col='id',nrows=1000)
    test_dataframe = pd.read_csv("ci/NYC_taxi_trip_duration/input/test1000.csv", index_col='id',nrows=1000)

    data_preprocessor = NYCTaxisPreprocessor(["trip_duration"])
    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)

    #test if there are any NaNs
    assert(X_train.isnull().sum().sum() == 0)
    assert(X_test.isnull().sum().sum() == 0)
    
    #test the outputs have the correct shapes
    assert X_train.shape == (1000,10), "X_train.shape should be {}, and is {}.".format((1000,10), X_train.shape)
    assert X_test.shape == (1000,9), "X_test.shape should be {}, and is {}.".format((1000,9), X_test.shape)
    assert y_train.shape == (1000,2), "y_train.shape should be {}, and is {}.".format((1000,2), y_train.shape)

#%% function called automaitcally by pytest to execute the tests
def test_taxis_feature_selector():
    train_dataframe = pd.read_csv('ci/NYC_taxi_trip_duration/input/train1000.csv', index_col='id',nrows=1000)
    test_dataframe = pd.read_csv("ci/NYC_taxi_trip_duration/input/test1000.csv", index_col='id',nrows=1000)
    train_dataframe["log_duration"] = np.log(train_dataframe.trip_duration)

    data_preprocessor = NYCTaxisFeatureSelector(["trip_duration", "log_duration"])
    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)

    #test the outputs have the correct shapes
    assert X_train.shape == (1000,4), "X_train.shape should be {}, and is {}.".format((1000,4), X_train.shape)
    assert X_test.shape == (1000,4), "X_test.shape should be {}, and is {}.".format((1000,4), X_test.shape)
    assert y_train.shape == (1000,1), "y_train.shape should be {}, and is {}.".format((1000,1), y_train.shape)
