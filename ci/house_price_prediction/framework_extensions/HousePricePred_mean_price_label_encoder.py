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

#%% Class to assign the mean value of the predicted column to each label of a cat. feature

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
        return self
            
    def transform(self, col):
        new_col = col.copy()
        for cat, o in self.means.items():
            new_col.loc[col == cat] = o
        return new_col
        
#%% Implementing class with our one version of the preprocessing methods

class Mean_Price_Preprocessor(AbstractPreprocessor):
    # replace missing labels with the most common one (only when the NaNs are less than 10% of the full dataset)
    def _calc_missing_cat_replacements(self, col):
        if col.count() / col.shape[0] > 0.90:
            return col.value_counts().index[0]
        return 'NaN'

    # replace missing numerical values with the mean  
    def _calc_missing_num_replacements(self, col):
        return col.mean()
    
    # generate attribute that will store all the encoders in a dictionary
    def _set_cat_col_encoder(self, df):
        self.cat_encoders = {}
        for col_name in self.categorical_features:
            encoder = mean_pred_encoder(self.get_init_cols_to_predict()[0], col_name)
            self.cat_encoders[col_name] = encoder.fit(df.fillna('NaN'))
    
    # convert every feature to numeric (necessary to train regression models later)
    def _feat_eng(self, dataframe):
        for col_name in dataframe.select_dtypes(exclude = [np.number]):
            dataframe[col_name] = pd.to_numeric(dataframe[col_name])
            
    # Create a column with the log of the predict column
    def _feat_eng_train(self, dataframe):        
        for col_name in self.get_init_cols_to_predict():
            dataframe["Log_" + col_name] = np.log(dataframe[col_name])
            
            #don't forget to add it to the list of columns not present in the test set!
            self.append_cols_to_predict("Log_" + col_name)
            
    # set of dependent variables is only the last one (the one added in feat. eng.)
    def get_y(self, df):
       return df.loc[:, self.get_cols_to_predict()[-1]]
    