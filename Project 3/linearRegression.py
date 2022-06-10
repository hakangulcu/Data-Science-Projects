import matplotlib.pyplot as plot
import numpy as np
import random

#importing test data
f = open('./test1.txt', 'r')
#taking them into normal array
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

#converting data to numpy arrays
inputsTest = np.array(inputs, dtype=float)
outputsTest = np.array(outputs, dtype=float)
f.close()

#importing train data
f = open('./train1.txt', 'r')
#taking them into normal array
inputs = []
outputs = []

for i in f:
    i = i.strip().split('\t')
    inputs.append(i[0])
    outputs.append(i[1])

#converting data to numpy arrays
inputsTrain = np.array(inputs, dtype=float)
outputsTrain = np.array(outputs, dtype=float)
f.close()

#Y = a + bX is the general formula
#e is for summation sign and x is inputsTrain and y is outputsTrain
# a = e(y*x^2) - (x*y^2) / ne^x2 - ex^2
# b = ne(xy)-exy / nex^2 - ex2

inputSize = len(inputsTrain)

#first take mean of datasets
inputMean = np.mean(inputsTrain)
outputMean = np.mean(outputsTrain)

#second find the sum of y*x and x*x 
sumXX = np.sum(pow(inputsTrain,2))
sumYX = np.sum(outputsTrain * inputsTrain)

#third find the deviation of y*x and x*x 
deviationOfXX = sumXX - (inputSize * inputMean * inputMean)
deviationOfYX = sumYX - (inputSize * inputMean * outputMean)

#fourth calculating the regressors and finalizing formula
b0 = deviationOfYX / deviationOfXX
b1 = outputMean - ((deviationOfYX / deviationOfXX) * inputMean)

#prediction and calculating total lose
prediction = b0 * inputsTrain + b1
totalLose = sum(pow((prediction - outputsTrain), 2)) 

#plot of results
plot.scatter(inputsTrain, outputsTrain, label = "Data points") #points
plot.plot(inputsTrain, prediction, label = "Predicted") #prediction line
plot.xlabel('Inputs')
plot.ylabel('Outputs')
plot.legend()

plot.title("Total linear regression loss =  " + str(totalLose))
plot.show()

