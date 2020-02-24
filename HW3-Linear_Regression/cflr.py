import numpy as np
import random
import math
from sklearn.model_selection import train_test_split


def parseData(dataPath):
	full_data = np.genfromtxt(dataPath, delimiter=',')

	# ignore the header row and index column
	parsed_data = full_data[1:, 1:]

	#separates the target value from the rest of the data
	target_values = parsed_data[:, 0]
	features = parsed_data[:,1:]
	return features, target_values


#compute theta = (xT * x)^-1 (xT * y)
def weights(x,y):
	x_xT_inv = np.linalg.inv(np.dot(x.T, x))
	xT_y = np.dot(x.T, y)
	theta = np.dot(x_xT_inv, xT_y)
	
	return theta

#sqrt( (1/total observations) * summation from 1 to N: (Yi-Yi')^2) Yi is the Ytrain at i. Yi' is g(x)_i, the predicted value at i
def rmse(gx, yTest):
	se = math.pow((sum(yTest) - sum(gx)),2)
	mse = se/np.size(gx, axis=0)
	RMSE = math.sqrt(mse)
	
	return RMSE


def closedFormLinReg(data):

	x,y = parseData(data)
	#split into training and testing arrays. 2/3 is training
	xTrain, xTest, yTrain, yTest = np.asarray(train_test_split(x, y, train_size = 0.7, random_state = 0))

	#add leading 1
	xTrain = np.insert(xTrain, 0, 1, axis=1)
	xTest = np.insert(xTest, 0,1, axis=1)

	theta = weights(xTrain,yTrain)

	gx = np.dot(xTest,theta ) #shape is 14x1

	RMSE = rmse(gx, yTest)
	print("Root Mean Squared Error: ",RMSE)




if __name__ == "__main__":
    # execute only if run as a script
	closedFormLinReg('x06Simple.csv')