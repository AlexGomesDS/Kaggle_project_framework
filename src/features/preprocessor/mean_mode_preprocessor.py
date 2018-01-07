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

    # replace missing labels with the most common one (only when the NaNs are less than 10% of the full dataset)
    def _calc_missing_cat_replacements(self, col):
        if col.count() / col.shape[0] > 0.90:
            return col.value_counts().index[0]
        return 'NaN'
    
    # For categorical features, replace nan with the string "NaN" and apply labelencoding 
    #(also add the value Nan in case it appears in the test set and not in the train)
    def _gen_cat_col_encoder(self, col_name, df=pd.DataFrame()):
        col_with_NaN = df[col_name].fillna('NaN').append(pd.DataFrame(['NaN']))
        return LabelEncoder().fit(col_with_NaN.values.ravel())
    
    # Check if this is training set by looking for the output column called "SalePrice"
    def _is_trainning_set(self, dataframe):
        for col in self.get_cols_to_predict():
            if col not in dataframe.columns:
                return False
        return True
        #return self.get_col_to_predict() in dataframe.columns
    
    # Don't create any new features for both datasets
    def _feat_eng(self, dataframe):
        pass
    
    # Create a column with the log of the saleprice
    def _feat_eng_train(self, dataframe):
        cols_to_apply_log = list(self.get_cols_to_predict())
        for col_name in cols_to_apply_log:
            dataframe["Log_" + col_name] = np.log(dataframe[col_name])
            self.append_cols_to_predict("Log_" + col_name)


#%% Testing

if __name__ == '__main__':
    data_preprocessor = Mean_Mode_Preprocessor(["SalePrice"])
    

    train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
    test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)
    #tests
    #test if there are any NaNs
    assert(X_train.isnull().sum().sum() == 0)
    assert(X_test.isnull().sum().sum() == 0)
    assert(X_train.select_dtypes(exclude = [np.number]).shape[1] == 0)
    assert(X_test.select_dtypes(exclude = [np.number]).shape[1] == 0)