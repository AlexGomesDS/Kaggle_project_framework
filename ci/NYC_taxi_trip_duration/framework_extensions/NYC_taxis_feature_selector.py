# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:10:13 2017

@author: Alexandre


This class implements the following preprocessing methods:
    - apply standardscaling to every numeric feature
    - select only some of the features
"""


#%% Import onyl the necessary libraries
import pandas as pd
import numpy as np
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor
from sklearn.preprocessing import StandardScaler

#%% Implementing class with our one version of the preprocessing methods

class NYCTaxisFeatureSelector(AbstractPreprocessor):
    # apply standardscaling to every numeric feature
    def _set_num_scaler(self, dataframe):
        self.num_scalers = StandardScaler().fit(dataframe[self.numerical_features])

    def get_y(self, df):
        return df[ [self._cols_to_predict[-1]] ]

    def _feat_eng(self, dataframe):
        columns_to_keep = [
        'pickup_hour',
        'pickup_weekday',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'log_duration']

        columns_to_drop = [col for col in dataframe.columns if col not in columns_to_keep]
        dataframe.drop(columns_to_drop, axis = 1,inplace=True)
    
