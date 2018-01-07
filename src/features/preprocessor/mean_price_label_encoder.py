# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 23:33:01 2018

@author: Alexandre

This class implements the following preprocessing methods:
    - Replace NaN values by their mean / most common label
    - Encode the categorical features according to the mean of the output for each category
    
"""


#%% Import onyl the necessary libraries
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 

import numpy as np
import pandas as pd

#%%

class mean_pred_encoder():
    def __init__(self, col_y_name, cat_col_name):
        self.feature_to_predict = col_y_name
        self.col_name = cat_col_name
    
    def fit(self, df):
        col = df[self.col_name].append(pd.DataFrame(['NaN']))[0]
        new_df = pd.DataFrame()
        new_df['vals'] = col.unique()
        new_df.index = new_df.vals
        new_df['mean'] = df[[self.col_name, self.feature_to_predict]].groupby(self.col_name).mean()[self.feature_to_predict]
        self.means = new_df.fillna(new_df.mean())['mean'].to_dict()
            
    def transform(self, col):
        new_col = col.copy()
        for cat, o in self.means.items():
            new_col.loc[col == cat] = o
        return new_col
        
#%% Implementing class with our one version of the preprocessing methods

class Mean_Price_Preprocessor(AbstractPreprocessor):
    
    # replace missing numerical values with the mean
    def _calc_missing_num_replacements(self, col):
        return col.mean()

    # replace missing labels with the most common one (only when the NaNs are less than 10% of the full dataset)
    def _calc_missing_cat_replacements(self, col):
        return 'NaN'
    
    def _gen_cat_col_encoder(self, col_name, df = pd.DataFrame()):
        col_encoder = mean_pred_encoder(self.get_init_cols_to_predict()[0], col_name)
        col_encoder.fit(df.fillna('NaN'))
        return col_encoder
    
    # Check if this is training set by looking for the output column called "SalePrice"
    def _is_trainning_set(self, dataframe):
        for col in self.get_init_cols_to_predict() + self.get_cols_to_predict():
            if col in dataframe.columns:
                return True
        return False

    # Create a column with the log of the saleprice
    def _feat_eng_train(self, dataframe):
        #use list() to create a new instance
        cols_to_apply_log = list(self.get_init_cols_to_predict())
        
        for col_name in cols_to_apply_log:
            dataframe["Log_" + col_name] = np.log(dataframe[col_name])
            dataframe.drop(col_name, 1, inplace = True)
            self.set_cols_to_predict(["Log_" + col_name])
            
    def _feat_eng(self, dataframe):
        for col_name in dataframe.select_dtypes(exclude = [np.number]):
            dataframe[col_name] = pd.to_numeric(dataframe[col_name])
            

#%% Testing

if __name__ == '__main__':
    data_preprocessor = Mean_Price_Preprocessor(["SalePrice"])
    
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
