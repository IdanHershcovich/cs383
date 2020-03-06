import math, random
import numpy as np
np.random.seed(0)
# np.set_printoptions(suppress=True)

def standardize(data):
   stand = (data)
   m = data.mean(axis=0)
   std = data.std(axis=0,ddof=1)

   stand = (stand-m) / std
   return stand

def standardizeWithTrain(data, tr_mean, tr_std):
	stand = (data)
	m = tr_mean
	std = tr_std
	stand = (stand-m) / std
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
	testing_data_label = testing_data[:, -1]
	testing_data = standardizeWithTrain(testing_data[:,:-1], training_data[:,:-1].mean(axis=0), training_data[:,:-1].std(axis=0,ddof=1))
	
	spam = []
	not_spam = []
	for i in range(np.size(training_stand,axis=0)):
		if(training_data[i,-1:]) == 1:
			spam.append(training_stand[i])
		else:
			not_spam.append(training_stand[i])

	spam = np.asarray(spam)
	not_spam = np.asarray(not_spam)
	return training_data, testing_data, testing_data_label, spam, not_spam

#likelihood
def gaussianDist(feature, obs):
	mean = feature.mean()
	std = feature.std(ddof=1)
	if std <= 0.0000000001:
		return 0.0000000001

	variance = math.pow(std,2)
	pdf = 1/((std*(math.sqrt(2*(math.pi)))))
	x_mean = math.pow((obs-mean),2)
	twovar = 2*variance
	power = (-1) * (x_mean/twovar)
	pdf *= 	math.exp(power)

	if pdf == 0:
		pdf = 0.000000001

	return pdf


def likelihood(dictionary,y_class, testing_data):
	for row in range(np.size(testing_data,axis=0)):
		for feature in range(np.size(y_class,axis=1)):
			dictionary[row] +=  math.log(gaussianDist(y_class[:,feature],testing_data[row,feature]))
	
def precision(tp,fp):
	p = tp/(tp+fp)
	print(str.format("Precision: {}", p))
	return p

def recall(tp,fn):
	r = tp/(tp+fn)
	print(str.format("Recall: {}", r))
	return r

def fMeasure(pr,re):
	f = 2*pr*re / (pr + re) 
	print(str.format("f-Measure: {}", f))
	return f

def accuracy(testing_data,true_p, true_n):
	acc = (1/np.size(testing_data, axis=0)) * (true_p+true_n)
	print(str.format("accuracy: {}", acc))
	return acc

def binClassification(testing_data,true_class, pred_class):
	true_p = 0
	true_n = 0
	false_p = 0
	false_n = 0
	
	for i in range(len(pred_class)):
		if true_class[i] == pred_class[i] and true_class[i] == 1:
			true_p +=1
		elif true_class[i] == pred_class[i] and true_class[i] == 0:
			true_n += 1
		elif true_class[i] != pred_class[i] and true_class[i] == 1:
			false_n +=1
		elif true_class[i] != pred_class[i] and true_class[i] == 0:
			false_p +=1
	return true_p, true_n, false_p, false_n

def compare(spam,n_spam):
	predicted_data= {}
	for i in range(len(spam)):
		
		if spam[i] > n_spam[i]:
			predicted_data[i] = 1
		else: 
			predicted_data[i] = 0
	return predicted_data


if __name__ == "__main__":
    # execute only if run as a script
	training_data, testing_data, testing_data_label, spam, n_spam = parseAndClassify('spambase.data')
	ns_likelihood = {
		i: math.log((np.size(n_spam,axis=0)/np.size(training_data, axis=0)))
			
            for i in range(np.size(testing_data,axis=0))
	}
	s_likelihood = {
		i:math.log((np.size(spam,axis=0)/np.size(training_data,axis=0)))
			
            for i in range(np.size(testing_data,axis=0))
	}
	likelihood(s_likelihood,spam, testing_data)
	likelihood(ns_likelihood, n_spam, testing_data)
	
	predicted_class = compare(s_likelihood, ns_likelihood)

	true_p, true_n, false_p, false_n = binClassification(testing_data, testing_data_label, predicted_class)

	acc = accuracy(testing_data, true_p, true_n)
	pr = precision(true_p, false_p)
	re = recall(true_p, false_n)
	f_m = fMeasure(pr, re)
