from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
from sklearn import datasets
class Policy():
    def __init__(self, config, obs_high, obs_low):
        #initialize each tm
        self.tm1 = TMRegressor(config['nr_of_clauses'], config['T'], config['s'], platform=config['device'], weighted_clauses=config['weighted_clauses'])
        self.tm2 = TMRegressor(config['nr_of_clauses'], config['T'], config['s'], platform=config['device'], weighted_clauses=config['weighted_clauses'])
        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer(obs_high, obs_low)

    def init_binarizer(self, obs_high, obs_low):
        #create a list of lists of values?
        self.binarizer.transform(vals)

    def update(self, tm_1_input, tm_2_input):
        # take a list for each tm that is being updated.
        self.tm1.fit(tm_1_input['observations'], tm_1_input['target_q_vals'])
        self.tm2.fit(tm_2_input['observations'], tm_2_input['target_q_vals'])

    def predict(self, obs):
        #binarize input
        b_obs = self.binarizer.fit(obs)
        #pass it through each tm
        tm1_q_val = self.tm1.predict(b_obs)
        tm2_q_val = self.tm2.predict(b_obs)
        #return the q_vals np.array([tm1, tm2])
        return np.transpose(np.array([tm1_q_val, tm2_q_val]))
