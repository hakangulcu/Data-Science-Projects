"""
Hakan Gulcu
21702275
GE461 Project 2

To achieve this assignment, I will mostly use mostly sklearn library which has functions for both PCA and LDA.

Question 2
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import scipy.io as loader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

"""
Loadmat is a part of scipy.io external library to load .mat files in python.
Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
"""
digits = loader.loadmat('digits.mat')
labels = digits['labels']
features = digits['digits']

"""
As stated in assignment, we should create seperate subsets for training and testing.
train_test_split function of sklearn.model_selection is making this job.
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None) is the documentation

I would gave 30 percent to test size make model better but 50 percent is stated.
"""

#by assigning an integer to random_state, we are guaranteed that each run will be equal.
x_trainData,x_testData,y_trainData,y_testData = train_test_split(features, labels, test_size=0.5, train_size=0.5, random_state=0, shuffle=True)
y_trainData_Transposed = y_trainData.T
y_testData_Transposed = y_testData.T
"""
sklearn library has a function called discriminant_analysis to achieve LDA 
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
class sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
"""

ldaResult = LinearDiscriminantAnalysis(n_components=9)
ldaResult.fit(x_trainData, y_trainData_Transposed[0])
bases = ldaResult.scalings_
dimensions = 9

a=5
b=2

for i in range(dimensions):
    plot.axis("off")
    plot.subplot(a, b, i+1)
    plot.imshow((bases[:,i]).reshape(20,20))
    
plot.suptitle("Bases")
plot.show()

#################################################
"""
I choose component count as 9 because it is stated that is max
In each iteration, i transform train and test date and fit to gaussian classfier. 
After that, i make prediction and find accuracy by using predicted trainX and trainX. 
Finally, i assign them into np arrays 
"""

component_count = 9
train = np.zeros((9,2))
test = np.zeros((9,2))

for i in range(component_count):
    i = i+1
    ldaResult = LinearDiscriminantAnalysis(n_components=i).fit(x_trainData, y_trainData_Transposed[0])
    
    trainX_transformed = ldaResult.transform(x_trainData)
    testX_transformed = ldaResult.transform(x_testData)
   
    gaussian = GaussianNB() #gaussion classfier for each subspace
    gaussian.fit(trainX_transformed, (y_trainData_Transposed)[0]) #y_trainData fitted 

    #Train X Part
    trainX_prediction = gaussian.predict(trainX_transformed) #prediction of projected trainX
    trainX_accuracy = metrics.accuracy_score((y_trainData_Transposed)[0], trainX_prediction)
    trainX_error = 1 - trainX_accuracy #error rate

    #Test X part
    testX_prediction = gaussian.predict(testX_transformed) #prediction of projected testX
    testX_accuracy = metrics.accuracy_score((y_testData_Transposed)[0], testX_prediction)
    testX_error = 1 - testX_accuracy

    #Storing in arrays
    train[i-1, 1] = trainX_error #Classification error rate in y label
    train[i-1, 0] = i #at i'th component in x label
    test[i-1, 1] = testX_error #Classification error rate in y label
    test[i-1, 0] = i #at i'th component in x label

#print(component_count) 
plot.plot(train[:, 0], train[:, 1], label="training") #0 for component count, 1 for classification errors
plot.plot(test[:, 0], test[:, 1], label="test") #0 for component count, 1 for classification errors
plot.legend()
plot.title("Training vs Test")
plot.xlabel("Dimension of Each Subspace")
plot.ylabel("Classification Error")
plot.show()

plot.plot(train[:, 0], train[:, 1], label="training") #0 for component count, 1 for classification errors
plot.title("Training Set Results")
plot.xlabel("Dimension of Each Subspace")
plot.ylabel("Classification Error")
plot.show()

plot.plot(test[:, 0], test[:, 1], label="test")
plot.title("Test Set Results")
plot.xlabel("Dimension of Each Subspace")
plot.ylabel("Classification Error")
plot.show()
