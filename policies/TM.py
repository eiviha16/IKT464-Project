from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
import numpy as np
from sklearn import datasets
import random

class Policy():
    def __init__(self, config):
        #initialize each tm
        self.tm1 = TMRegressor(config['nr_of_clauses'], config['T'], config['s'], platform=config['device'], weighted_clauses=config['weighted_clauses'], min_y=config['y_min'], max_y=config['y_max'])
        self.tm2 = TMRegressor(config['nr_of_clauses'], config['T'], config['s'], platform=config['device'], weighted_clauses=config['weighted_clauses'], min_y=config['y_min'], max_y=config['y_max'])
        self.vals = np.loadtxt('./misc/observation_data.txt', delimiter=',').astype(dtype=np.float32)

        self.binarizer = StandardBinarizer(max_bits_per_feature=config['bits_per_feature'])
        self.init_binarizer()
        self.init_TMs()

    def init_binarizer(self):
        #create a list of lists of values?
        self.binarizer.fit(self.vals)

    def init_TMs(self):
        vals = self.binarizer.transform(self.vals)
        self.tm1.fit(vals[:100], np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]))
        self.tm2.fit(vals[:100],  np.array([random.randint(0, 60) for _ in range(len(vals[:100]))]))

    def update(self, tm_1_input, tm_2_input):
        # take a list for each tm that is being updated.
        self.tm1.fit(np.array(tm_1_input['observations']), np.array(tm_1_input['target_q_vals']))
        self.tm2.fit(np.array(tm_2_input['observations']), np.array(tm_2_input['target_q_vals']))

    def predict(self, obs):
        #binarize input
        if obs.ndim == 1:
            obs = obs.reshape(1, 4)
        b_obs = self.binarizer.transform(obs)
        #pass it through each tm
        tm1_q_val = self.tm1.predict(b_obs)
        tm2_q_val = self.tm2.predict(b_obs)
        #return the q_vals np.array([tm1, tm2])
        return np.transpose(np.array([tm1_q_val, tm2_q_val]))
