"""
Hakan Gulcu
21702275
GE461 Project 2

To achieve this assignment, I will mostly use mostly sklearn library which has functions for both PCA and LDA.

Question 1 
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plot
#%matplotlib inline
import scipy.io as loader
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

"""
sklearn library has a function called decomposition to achieve PCA 
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
"""

pcaResult = PCA(n_components=400) #project the 400-dimensional data
pcaResult.fit_transform(x_trainData)
pcaComponents = pcaResult.components_
pcaEigenVectors = pcaResult.explained_variance_
pcaMean = pcaResult.mean_
pcaPercentageOfVariance = pcaResult.explained_variance_ratio_
print("PCA calculations end.")

#Question 1.1

plot.plot(pcaEigenVectors)
plot.xlabel("Components")
plot.ylabel("Eigen Values")
plot.title("Eigen Values")
plot.show() #plot eigen values in descending order

#Question 1.2

#each pattern digitized 20x20 so we need to reshape
x_trainDataMean = pcaResult.mean_.reshape(20,20)
image1 = plot.imshow(x_trainDataMean)
plot.title("X Train Data Mean Not Transposed")
plot.show()

#when we reshape directly, the image is not appropriete so we need to transpose it to right form
x_trainDataMeanTransposed = pcaResult.mean_.reshape(20,20).T #.T transpose
image2 = plot.imshow(x_trainDataMeanTransposed)
plot.title("X Train Data Mean Transposed")
plot.show()

#by looking this we can decide component number where it becomes straight, 
#when i looked it, i decided to choose as 60.
print(pcaPercentageOfVariance.cumsum()) 

a = 6  #interval
b = 10 #times
for i in range(a*b):
    plot.axis("off")
    plot.subplot(a, b, i+1)
    plot.imshow(pcaComponents[i].reshape(20,20).T) #20x20

plot.suptitle("First 60 Components")
plot.show()    

#Question 1.3

"""
I choose component count as 200 because it is stated the more is better.
In each iteration, i transform train and test date and fit to gaussian classfier. 
After that, i make prediction and find accuracy by using predicted trainX and trainX. 
Finally, i assign them into np arrays 
"""

component_count = 200
train = np.zeros((200,2))
test = np.zeros((200,2))

for i in range(component_count):
    i = i+1
    pcaResult = PCA(n_components = i, random_state=0).fit(x_trainData) #for each iteration PCA for i

    trainX_transformed = pcaResult.transform(x_trainData)
    testX_transformed = pcaResult.transform(x_testData)
   
    gaussian = GaussianNB() #gaussion classfier for each subspace
    gaussian.fit(trainX_transformed, (y_trainData.T)[0]) #y_trainData fitted 

    #Train X Part
    trainX_prediction = gaussian.predict(trainX_transformed) #prediction of projected trainX
    trainX_accuracy = metrics.accuracy_score((y_trainData.T)[0], trainX_prediction)
    trainX_error = 1 - trainX_accuracy #error rate

    #Test X part
    testX_prediction = gaussian.predict(testX_transformed) #prediction of projected testX
    testX_accuracy = metrics.accuracy_score((y_testData.T)[0], testX_prediction)
    testX_error = 1 - testX_accuracy

    #Storing in arrays
    train[i-1, 1] = trainX_error #Classification error rate in y label
    train[i-1, 0] = i #at i'th component in x label
    test[i-1, 1] = testX_error #Classification error rate in y label
    test[i-1, 0] = i #at i'th component in x label

#print(component_count) 
plot.plot(train[:, 0], train[:, 1], label="training") #0 for component count, 1 for classification errors
plot.plot(test[:, 0], test[:, 1], label="test")
plot.legend()
plot.title("Training vs Test")
plot.xlabel("Number of Components")
plot.ylabel("Classification Error")
plot.show()

plot.plot(train[:, 0], train[:, 1], label="training") #0 for component count, 1 for classification errors
plot.title("Training Set Results")
plot.xlabel("Number of Components")
plot.ylabel("Classification Error")
plot.show()

plot.plot(test[:, 0], test[:, 1], label="test")
plot.title("Test Set Results")
plot.xlabel("Number of Components")
plot.ylabel("Classification Error")
plot.show()
