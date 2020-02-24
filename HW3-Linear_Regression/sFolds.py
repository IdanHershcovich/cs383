import numpy as np
import random
import math

from cflr import weights

np.set_printoptions(suppress=True)
np.random.seed(0)

def parseData(dataPath):
	full_data = np.genfromtxt(dataPath, delimiter=',')

	# ignore the header row and index column
	parsed_data = full_data[1:, 1:]
	#separates the target value from the rest of the data
	target_values = parsed_data[:, 0]
	features = parsed_data[:,1:]
	return parsed_data
	# return features, target_values

#squared error for 1 value at a time
def squaredError(gx, yTest):
	se = pow((yTest - gx),2)
	return se
	
def sFolds(data, S):
	data = parseData(data)

	#defines how many elements per fold
	stopper = math.ceil(len(data) / S)	
	RMSE = []
	
	for runs in range(20):
		
		np.random.shuffle(data)
		folds = []
		se = []

		#temp var so I can delete from X without altering original data
		temp = data[:]

		#Appending from 0 to S and then deleting that section. Repeat S times. Folds will have S arrays, each of size S.
		for K in range(S):
			folds.append(temp[0:stopper])
			temp = np.delete(temp,slice(0,stopper), axis=0)
	
	
		for i in range(S):
			
			#Set the folds to a temp var. Get the first fold, set it as 
			temp = folds[:]
			testing_data = np.asarray(temp[i])
			del temp[i]
			training_data = np.asarray(temp)		
			training_data = np.concatenate([training_data[f] for f in range(len(training_data))],axis=0)
			
			#Parsing the test and train data into features and target
			yTest = testing_data[:,0]
			xTest = testing_data[:,1:]
			yTrain = training_data[:,0]
			xTrain = training_data[:,1:]
			

			# add leading 1
			xTrain = np.insert(xTrain, 0, 1, axis=1)
			xTest = np.insert(xTest, 0,1, axis=1)

			theta = weights(xTrain,yTrain)

			# Applying the closed form to the testing data
			gx = np.dot(xTest,theta)
			
			# Computing all the square errors for each sample in the testing target
			for sample in range(len(testing_data)):
				se.append(squaredError(gx[sample], yTest[sample]))
		#Computing the RMSE. One for each of the 20 runs
		RMSE.append(math.sqrt(sum(se)/len(se)))
		
	RMSE = np.asarray(RMSE)
	
	# Return the mean and std
	print(str.format("When S = {}, Mean RMSE = {} and the Standard Deviation is = {}", S, RMSE.mean(), RMSE.std()))

	

sFolds('x06Simple.csv',2)
sFolds('x06Simple.csv',4)
sFolds('x06Simple.csv',22)
sFolds('x06Simple.csv',44)


