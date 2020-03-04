import math, random
import numpy as np
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

def gaussianDist(vector, feature):
	e = math.e
	pdf = 1/((vector.std())*(math.sqrt(2*(math.pi))))
	x_mean = math.pow((feature-vector.mean()),2)
	twovar = 2*vector.var()
	power = (-1) * (x_mean/twovar)
	# power = (-1)*((math.pow((feature-vector.mean()),2))/(2*(feature.var())))
	pdf = pdf * math.pow(e,power)

	return pdf

def bayes(data):	

	training_data,testing_data, spam, n_spam = parseAndClassify(data)
	spam = np.asarray(spam)
	n_spam = np.asarray(n_spam)
	# spam_model = []
	n_spam_model = {}
	spam_model = {}

	# dictionary for a gaussian model for each feature for each class. each dict has a singular value for each feature
	for feature in range(np.size(spam,axis=1)):
		for row in range(np.size(spam,axis=0)):
			spam_model["feature{0}".format(feature+1)] = gaussianDist(spam,spam[row,feature])

	for feature in range(np.size(n_spam,axis=1)):
		for row in range(np.size(n_spam,axis=0)):
			n_spam_model["feature{0}".format(feature+1)] = gaussianDist(n_spam,n_spam[row,feature])
	import pdb; pdb.set_trace()

bayes('spambase.data')
