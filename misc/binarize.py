from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
import numpy as np
data = np.loadtxt('../scrapbook/data/observations4.txt', delimiter=',').astype(dtype=np.float32)
b = StandardBinarizer(max_bits_per_feature=1000)
data_transformed = b.fit_transform(data)
a = 2