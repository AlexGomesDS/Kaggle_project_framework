# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:10:13 2017

@author: Alexandre


This class implements the following preprocessing methods:

    - rescale every numerical feature with StandardScaler (0 mean and std rescaled)
    - create new features pickup hour and weekday
    - apply logarithm to the trip duration
"""


#%% Import onyl the necessary libraries

from features.preprocessor.abstract_preprocessor import AbstractPreprocessor
import pandas as pd
import numpy as np
from datetime import datetime

#%% Implementing class with our one version of the preprocessing methods

class NYCTaxisPreprocessor(AbstractPreprocessor):

    # override list of categorical features
    def _set_cat_fetures(self, df):
        self.categorical_features=[]

    # don't replace nan's because there shouldn't be none
    def _set_missing_replacements(self, df):
        self.NaN_replacements = None

    # remove unused columns vendor_id, store_and_fwd_flag, passenger_count
    def _feat_eng(self, dataframe):
        #drop some unnecessary columns if they exist
        columns_to_be_dropped = ['vendor_id', 'passenger_count', 'store_and_fwd_flag']
        dataframe.drop(columns_to_be_dropped, axis=1, inplace = True)
        #Create time features (hour and weekday)
        dataframe['pickup_datetime'] = dataframe.pickup_datetime.apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        dataframe['pickup_weekday'] = dataframe.pickup_datetime.apply(datetime.weekday)
        dataframe['pickup_hour'] = dataframe.pickup_datetime.apply(lambda x: x.hour)

        #calculate rescaled distances (Euclidean + Manhattan)
        dataframe['Eucl_distance'] = np.sqrt(
                                    (dataframe.dropoff_latitude - dataframe.pickup_latitude).pow(2) +
                                    (dataframe.dropoff_longitude - dataframe.pickup_longitude).pow(2) )

        dataframe['Manh_distance'] = ( (dataframe.dropoff_latitude - dataframe.pickup_latitude).abs() +
                                (dataframe.dropoff_longitude - dataframe.pickup_longitude).abs())

    # apply to the predicting column
    def _feat_eng_train(self, dataframe):
        dataframe['dropoff_datetime'] = dataframe.dropoff_datetime.apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        dataframe['log_duration'] = np.log(dataframe.trip_duration)
        self.append_cols_to_predict('log_duration')
