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

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

#%% Preprocess the data sets
class AbstractPreprocessor(AbstractFeatures):
    __metaclass__ = ABCMeta

    def __init__(self, output_cols):
        if isinstance(output_cols, list):
            self._cols_to_predict = output_cols
        else:
            self._cols_to_predict = [output_cols]

        # store list of columns in the dataset
        self._list_of_features = None

        #default NaN replacements
        self.NaN_replacements = None

        # default categorical features NaN replacement
        self.cat_missing_replacement = None

        # default numeric features NaN replacement
        self.num_missing_replacement = None

        # initialize categorical features encoder
        self.cat_encoders = None

        # initialize numerical features scaler
        self.num_scalers = None


    #### Methods that should be overwritten when extending this class
	# given a dataframe column of a numerical feature, return the value we want to replace the NaN's with
    def _calc_missing_num_replacements(self, col):
        return None


    # given a dataframe column of a categorical feature, return the value we want to replace the NaN's with
    def _calc_missing_cat_replacements(self, col):
        return None

    # Generate an encoder for a given column for a categorical feature, store it in the dictionary "self.cat_encoders"
    def _set_cat_col_encoder(self, dataframe):
        pass


    # Generate a Feature scaler, store it in the dictionary "self.num_scalers"
    def _set_num_scaler(self, dataframe):
        pass


    # Feature engineering (like adding columns). Apply all the changes to the dataframe
    def _feat_eng(self, dataframe):
        pass


    # Feature engineering to be applied only to the train set
    def _feat_eng_train(self, dataframe):
        pass


    #### Methods that shouldn't need to be overwritten
    # get set of independent variables
    def get_X(self, df):
        return df[[col for col in df.columns if col not in self._cols_to_predict]]


    # get set of dependent variables
    def get_y(self, df):
        return df[self._cols_to_predict]


    # method to check if a given dataframe is for training or testing
    def _is_trainning_set(self, dataframe):
        for col in self._cols_to_predict:
            if col in dataframe.columns:
                return True
        return False


	 # method to build dataframe with a replacement value to the NaN's, per column, to be used later in the cook method
    def _set_missing_replacements(self, df):
        list_of_col_replacements = [
                self._calc_missing_cat_replacements(df[c])
            if c in self.categorical_features else
                self._calc_missing_num_replacements(df[c])
            for c in df]
        
        # for the following method to work all columns must have a valid NaN replacement
        for val in list_of_col_replacements:
            if val is None:
                return
        
        self.NaN_replacements = pd.Series(list_of_col_replacements, index=df.columns)


    # method to replace the NaN's in a dataframe by the values determined in _set_missing_replacements()
    def _apply_missing_replacements(self, df):
        df.fillna(self.NaN_replacements, inplace = True)


    # Apply the encoding prepared for the categorical features
    def _apply_cat_col_encoding(self, dataframe):
        for col_name in self.categorical_features:
            column_encoder = self.cat_encoders[col_name]
            dataframe[col_name] = column_encoder.transform(dataframe[col_name])


    # Feature scaling
    def _apply_num_scaler(self, dataframe):
        dataframe[self.numerical_features] = self.num_scalers.transform(dataframe[self.numerical_features])


    # method to set the list of features categorical
    def _set_cat_features(self, training_data_frame):
        self.categorical_features   = [c for c in training_data_frame
            if is_string_dtype(training_data_frame[c]) and
                c not in self._cols_to_predict and
                c in self._list_of_columns]


    # method to set the list of features numerical
    def _set_num_features(self, training_data_frame):
        self.numerical_features     = [c for c in training_data_frame
            if is_numeric_dtype(training_data_frame[c]) and
                c not in self._cols_to_predict and
                c in self._list_of_columns]


    #### Framework basic structure, i.e these methods shouldn't be overwritten
    # get every column to be predicted (even the ones added eg LogSalePrice)
    def get_cols_to_predict(self):
        return self._cols_to_predict


    # get every column to be predicted (even the ones added eg LogSalePrice)
    def get_init_cols_to_predict(self):
        return [col for col in self._cols_to_predict if col in self._list_of_columns]


    # append column to list of columns not present in the test set. Necessary for the X,y split
    def append_cols_to_predict(self, new_col):
        if new_col not in self._cols_to_predict:
            self._cols_to_predict.append(new_col)


    #apply cook and then separate the resulting dataset in two: X and y
    def cook_and_split(self, df):
        df_eng = self.cook(df)
        if self._is_trainning_set(df):
            #returning X and y separately for the training set
            return self.get_X(df_eng), self.get_y(df_eng)
        else:
            return df_eng, pd.DataFrame()


    # execute both prepare and cook, to use with the train set
    def prepare_and_cook(self, training_data_frame):
        self.prepare(training_data_frame)
        return self.cook(training_data_frame)


    # train the preprocessor using the data from the training set
    def prepare(self, training_data_frame):
        # set list of all features
        self._list_of_columns = training_data_frame.columns

        # create list of categorical and numerical features
        self._set_cat_features(training_data_frame)
        self._set_num_features(training_data_frame)

        # Prepare replacements to be applied to missing values
        self._set_missing_replacements(training_data_frame)

        # Prepare label encoding (we'll need one encoder per column)
        self._set_cat_col_encoder(training_data_frame)

        # Prepare numeric features scaling
        self._set_num_scaler(training_data_frame)
        
        return self

    #Apply the preprocessing methods prepared with a certain dataset, to any given dataset
    def cook(self, df):
        #initialize engineered D.F. (must use a copy otherwise we would affect the source variable)
        df_eng = df.copy()

        #Replace missing values by the ones we chose
        if self.NaN_replacements is not None:
            self._apply_missing_replacements(df_eng)

        # encode categorical features
        if self.cat_encoders is not None:
            self._apply_cat_col_encoding(df_eng)

        # re-scale numerical features
        if self.num_scalers is not None:
            self._apply_num_scaler(df_eng)

        # feature engineering
        self._feat_eng(df_eng)

        # predicted features engineering
        if self._is_trainning_set(df_eng):
            self._feat_eng_train(df_eng)

        return df_eng
