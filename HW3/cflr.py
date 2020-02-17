import numpy as np
import random
import math
from sklearn.model_selection import train_test_split


np.set_printoptions(suppress=True)
random.seed(0)


def parseData(dataPath):
	full_data = np.genfromtxt(dataPath, delimiter=',')

	# ignore the header row and index column
	parsed_data = full_data[1:, 1:]
	#separates the target value from the rest of the data
	target_values = parsed_data[:, 0]
	features = parsed_data[:,1:]

	

	return features, target_values
	# return parsed_data


#compute theta = (x * xT)^-1 (xT * y)
def weights(x,y):
	x_xT_inv = np.linalg.inv(np.dot(x, x.T))
	xT_y = np.dot(x.T, y)
	theta = np.dot(x_xT_inv, xT_y)
	print(theta)
	return theta

def closedFormLinReg(data):

	x,y = parseData(data)
	#get training data (2/3)
	# entries_in_data = np.size(data, axis=0)

	#split into training and testing arrays. 2/3 is training
	xTrain, xTest, yTrain, yTest = np.asarray(train_test_split(x, y, train_size = 0.7, random_state = 0))
	print("xtrain size {}: ", len(xTrain))
	print("xTest size {}: ", len(xTest))
	print("yTrain size {}: ", len(yTrain))
	print("yTest size {}: ", len(yTest))

	#add leading 1
	xTrain = np.insert(xTrain, 0, 1, axis=1)

	theta = weights(xTrain,yTrain)
	






	

	




closedFormLinReg('x06Simple.csv')