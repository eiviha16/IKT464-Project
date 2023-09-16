from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
from sklearn import datasets
class Policy():
    def __init__(self, config):
        #initialize each tm
        tm1 = TMRegressor()
        tm2 = TMRegressor()
    def train_tms(self, tm_1_input, tm_2_input):
        # take a list for each tm that is being updated.
        self.tm1.fit(tm_1_input['observations'], tm_1_input['target_q_vals'])
        self.tm2.fit(tm_2_input['observations'], tm_2_input['target_q_vals'])

    def predict(self, input):
        #binarize input

        #pass it through each tm

        #return the q_vals np.array([tm1, tm2])
        return q_vals
