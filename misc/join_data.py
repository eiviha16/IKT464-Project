import numpy as np

obs1 = np.loadtxt('../scrapbook/data/observations.txt', delimiter=',').astype(dtype=np.float32)
obs2 = np.loadtxt('../scrapbook/data/observations2.txt', delimiter=',').astype(dtype=np.float32)
obs3 = np.loadtxt('../scrapbook/data/observations3.txt', delimiter=',').astype(dtype=np.float32)
obs4 = np.loadtxt('../scrapbook/data/observations4.txt', delimiter=',').astype(dtype=np.float32)

observations = np.vstack((obs1, obs2, obs3, obs4))
np.savetxt('observation_data.txt', observations, delimiter=',', fmt='%f')
