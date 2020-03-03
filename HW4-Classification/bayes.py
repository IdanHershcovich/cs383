import math
import random

import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
np.set_printoptions(suppress=True)

def standardize(data):
   stand = (data)
   test = data.mean(axis=(0), keepdims=False)
   test2 = data.std(axis=(0), keepdims=False)

   stand = (stand-test) / test2
   return stand

def parseAndClassify(data):
	# get data from file
	full_data = np.genfromtxt(data, delimiter=',')
	# shuffle
	np.random.shuffle(full_data)

	# make a copy of the full data to make the separation of training (2/3) and testng (1/3) 
	temp = full_data[:]
	training_data = temp[0:math.ceil(np.size(temp,axis=0)*(2/3))]
	temp = np.delete(temp,slice(0,np.size(training_data,axis=0)), axis=0)
	testing_data = temp	

	#Separate into spam and not spam
	training_stand = standardize(training_data[:,:-1])
	spam = []
	not_spam = []
	for i in range(np.size(training_stand,axis=0)):
		if(training_data[i,-1:]) == 1:
			
			spam.append(training_stand[i])
		else:
			
			not_spam.append(training_stand[i])


	return training_data, testing_data, spam, not_spam


def bayes(data):	

	tr,te, spam, n_spam = parseAndClassify(data)
	
	
	import pdb; pdb.set_trace()
	


bayes('spambase.data')
