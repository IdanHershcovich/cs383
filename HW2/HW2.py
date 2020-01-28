import os
import numpy as np



data = np.genfromtxt('diabetes.csv', delimiter=',')

class_label = data[:,0]

obs_data = data[:,1:]

def standardize(data):
   stand = (data-data.mean(axis=(0), keepdims=False))/data.std(axis=(0,1), keepdims=False, ddof=1)
   return stand, data.mean(axis=(0), keepdims=False), data.std(axis=(0,1), keepdims=False, ddof=1)

def myKMeans(data_clusterX, target_clusterY, k)


print(str.format("shape of label {}", class_label.shape))
print(str.format("shape of obs_data {}", obs_data.shape))

