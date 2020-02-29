import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)

def standardize(data):
   stand = (data)
   test = data.mean(axis=(0), keepdims=False)
   test2 = data.std(axis=(0), keepdims=False)

   stand = (stand-test) / test2
   return stand


def bayes(data):
	full_data = np.genfromtxt(data, delimiter=',')
	features = full_data[:,:-1]
	label = full_data[:,-1]

	#split into training and testing arrays. 2/3 is training
	xTrain, xTest, yTrain, yTest = np.asarray(train_test_split(features, label, train_size = 0.7, random_state = 0))

	xTrain=standardize(xTrain)
	yTrain=standardize(yTrain)
	import pdb
	pdb.set_trace()

	


bayes('spambase.data')
