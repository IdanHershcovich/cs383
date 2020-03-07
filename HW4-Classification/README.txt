Both Files need to have the data in the same root directory! Both files need to be in the same place as well since logReg imports from bayes.

Bayes.py: Naive Bayes classifier program

This script separates a given dataset into training and testing, and then into spam and not spam class labels. It then makes a prediction based on Gaussian models and the prior for its
label. In order to test this prediction, we compute the precision, recall, f-measure and accuracy. 


LogReg.py: Logistic Regression

This script is similar to the bayes.py and they share some functionality as well.  It also takes in the dataset. It trains theta weights and then applies them to the testing data to predict 
if the class is spam or not. 

Possible issues with LogReg:
I found a math domain error once, but it never came up again.
The statistics range from 88.9-90.3