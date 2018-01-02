# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:10:13 2017

@author: Alexandre


This class implements the following preprocessing methods:
    - Replace NaN values by their mean / most common label
    - Label encoding of the categorical features
"""


#%% Import onyl the necessary libraries
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%% Implementing class with our one version of the preprocessing methods

class Mean_Mode_Preprocessor(AbstractPreprocessor):
    
    # replace missing numerical values with the mean
    def _calc_missing_num_replacements(self, col):
        return col.mean()

    # replace missing labels with the most common one
    def _calc_missing_cat_replacements(self, col):
        return col.value_counts().index[0]
    
    # For categorical features, replace nan with the string "NaN" and apply labelencoding 
    #(also add the value Nan in case it appears in the test set and not in the train)
    def _gen_cat_col_encoder(self, col):
        return LabelEncoder().fit(col.fillna('NaN').append(pd.DataFrame(['NaN'])))
    
    # Check if this is training set by looking for the output column called "SalePrice"
    def _is_trainning_set(self, dataframe):
        return "SalePrice" in dataframe.columns
    
    # Don't create any new features for both datasets
    def _feat_eng(self, dataframe):
        pass
    
    def _feat_eng_train(self, dataframe):
        pass
    
