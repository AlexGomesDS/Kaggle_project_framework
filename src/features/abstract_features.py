# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:43:15 2017

@author: ASSG
"""

from abc import ABCMeta
from abc import abstractmethod

class AbstractFeatures:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def prepare(self, input_data):
        return NotImplemented
    
    @abstractmethod
    def cook(self, input_data):
        return NotImplemented
    