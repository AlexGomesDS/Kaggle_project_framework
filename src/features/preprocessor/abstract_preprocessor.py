# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:06:25 2017

@author: Alexandre

This class implements a class for all our preprocessing + feature engineering.

The idea is to remove all that part from the main code in order to make it cleaner and easy to read.

This class should be extended and the abstract methods implemented with the preprocessing methods of your choice.
"""

#%% Import only the necessary libraries
from features.abstract_features import AbstractFeatures 
from abc import ABCMeta
from abc import abstractmethod

import pandas as pd
import numpy as np

#%% Preprocess the data sets
class AbstractPreprocessor(AbstractFeatures):
    __metaclass__ = ABCMeta
    
    def __init__(self, output_cols):
        if isinstance(output_cols, list):
            self._cols_to_predict = output_cols
        else:
            self._cols_to_predict = [output_cols]
        #since we may want to add columns from the one to predict we need to store the original ones
        self._init_cols_to_predict = list(self._cols_to_predict)

    @abstractmethod
    # given a dataframe column of a numerical feature, return the value we want to replace the NaN's with
    def _calc_missing_num_replacements(self, col):
        return NotImplemented
    
    @abstractmethod
    # given a dataframe column of a categorical feature, return the value we want to replace the NaN's with
    def _calc_missing_cat_replacements(self, col):
        return NotImplemented
    
    @abstractmethod
    # Generate an encoder for a given column for a categorical feature (must return an object from a class that implements a transform() method)
    def _gen_cat_col_encoder(self, col):
        return NotImplemented
    
    @abstractmethod
    # Check if this is training or test set
    def _is_trainning_set(self, dataframe):
        return NotImplemented
    
    @abstractmethod
    # Feature engineering (like adding columns)
    def _feat_eng(self, dataframe):
        return NotImplemented
    
    @abstractmethod
    # Feature engineering to be applied only to the train set
    def _feat_eng_train(self, dataframe):
        return NotImplemented
    
    # train the preprocessor using the data from the training set
    def prepare(self, training_data_frame):
        self.categorical_features   = [c for c in training_data_frame if training_data_frame[c].dtype == np.dtype('O')]
        self.numerical_features     = [c for c in training_data_frame if training_data_frame[c].dtype != np.dtype('O')]
        
        # Get replacements for NaN values: column mean for numeric and most common for categorical features
        self.NaN_replacements = pd.Series([
                self._calc_missing_cat_replacements(training_data_frame[c])
            if c in self.categorical_features else 
                self._calc_missing_num_replacements(training_data_frame[c])
            for c in training_data_frame],
            index=training_data_frame.columns)
        
        # Prepare label encoding (we'll need one encoder per column)
        self.cat_le = {}
        for col_name in self.categorical_features:
            self.cat_le[col_name] = self._gen_cat_col_encoder(col_name, training_data_frame)
            
    #Apply the preprocessing methods prepared with a certain dataset, to any given dataset
    def cook(self, df):
        #initialize engineered D.F. (must use a copy otherwise we would affect the source variable)
        df_eng = df.copy()
        
        #Replace missing values by the ones we chose
        df_eng.fillna(self.NaN_replacements, inplace = True)
        
        # encode categorical features
        for col_name in self.categorical_features:
            column_encoder = self.cat_le[col_name]
            df_eng[col_name] = column_encoder.transform(df_eng[col_name])
        
        #feature engineering
        self._feat_eng(df_eng)
        if self._is_trainning_set(df_eng):
            self._feat_eng_train(df_eng)

        return df_eng
    
    #apply cook and then separate the resulting dataset in two: X and y
    def cook_and_split(self, df):
        df_eng = self.cook(df)
        if self._is_trainning_set(df):
            #returning X and y separately for the training set
            return (df_eng.loc[:, [col for col in df_eng.columns if col not in self._cols_to_predict]], 
                    df_eng.loc[:, self._cols_to_predict])
        else:
            return df_eng, pd.DataFrame()
        
    def prepare_and_cook(self, training_data_frame):
        self.prepare(training_data_frame)
        return self.cook(training_data_frame)
    
    def get_cols_to_predict(self):
        return self._cols_to_predict
    
    def get_init_cols_to_predict(self):
        return self._init_cols_to_predict
    
    def set_cols_to_predict(self, new_cols_list):
        self._cols_to_predict = new_cols_list
