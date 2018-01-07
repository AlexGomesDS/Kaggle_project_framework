# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:16:10 2017

@author: Joana Pinto
"""

import pandas as pd

class SubmissionWriter:

    def build_submission(self, id, prediction, classes):
        submission_DF = pd.concat([id,
                                   pd.DataFrame(prediction, columns = classes)], 
                       axis = 1)
        return submission_DF
    
    def write_submission_to_csv(self, filename, id, predictions, classes):
        self.build_submission(id, predictions, classes).to_csv(filename, index = False)
        
#if __name__ == "__main__":
    