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

class ArtificialNeuralNetwork(object):
    def __init__(self, hiddenLayer):
        self.hiddenLayer = hiddenLayer
        self.hiddenLayerWeight = np.array([random.random() for i in range(hiddenLayer)]) 
        self.inputWeight = np.array([random.random() for i in range(hiddenLayer)]) 
        self.outputWeight = 1 / float(hiddenLayer)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-(x)))
    
    def derivativeSigmoid(self, sigm): 
        return sigm * (1-sigm)

    def errorCalculation(self, out, sum):
        return out - sum

    def createPlot(self, input, output):
        pred = self.prediction
        totalLose = 0
        totalLose = sum(pow((pred - output), 2)) 

        plot.scatter(input, output, label = "Data points") #points
        plot.scatter(input, pred, label = "Predicted") #prediction line
        plot.xlabel('Inputs')
        plot.ylabel('Outputs')
        plot.legend()
        plot.title("Total ANN loss = " + str(totalLose))
        plot.show()

    def train(self, input, output, epoch, lr):
        #print("I am there.")
        for i in range (epoch):
           
            index = np.random.randint(0,len(input))   
            hiddenValue = self.hiddenLayerWeight
            inputValue = self.inputWeight
            outputValue = self.outputWeight

            fx = inputValue + input[index] * hiddenValue 
            fxSigmoid = self.sigmoid(fx)
            fxSigmoidDerivative = self.derivativeSigmoid(fxSigmoid)
            errorF = self.errorCalculation(output[index], np.sum(fxSigmoid * outputValue))
            
            self.inputWeight = self.inputWeight + (lr * errorF * outputValue * fxSigmoidDerivative)
            self.outputWeight = self.outputWeight + (lr * errorF * fxSigmoid)
            self.hiddenLayerWeight = self.hiddenLayerWeight + (lr * errorF * outputValue * fxSigmoidDerivative * input[index])

            fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
            fxSigmoid = self.sigmoid(fxWithReshape)
            prediction = np.dot(fxSigmoid, outputValue)
            totalLose = 0
            totalLose = sum(pow((prediction - output.reshape(len(output),1)), 2))
        #print("For epoch = ", epoch , ", loss = ", totalLose)

    def predict(self, input, output, tp, tpValue):
        hiddenValue = self.hiddenLayerWeight
        inputValue = self.inputWeight
        outputValue = self.outputWeight
        hiddenLayerC = self.hiddenLayer
        fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
        fxSigmoid = self.sigmoid(fxWithReshape)
        self.prediction = np.dot(fxSigmoid, outputValue)
        pred = self.prediction
        totalLose = 0
        totalLose = sum(pow((pred - output), 2)) 
        print("For " + str(tpValue), tp, ", total loss =", totalLose)

    def calculateResults(self, input, output):
        a = 5
        b = 7
        hiddenValue = self.hiddenLayerWeight
        inputValue = self.inputWeight
        outputValue = self.outputWeight
        hiddenLayerC = self.hiddenLayer
        fxWithReshape = inputValue + input.reshape(len(input), 1) * hiddenValue
        fxSigmoid = self.sigmoid(fxWithReshape)
        self.prediction = np.dot(fxSigmoid, outputValue)
        pred = self.prediction
        totalLose = 0
        averageLoss = 0
        standartDerivation = 0
        totalLose = sum(pow((pred - output), 2)) 

        averageLoss = totalLose / len(input)
        standartDerivation = sum(pow((input - averageLoss),2))
        standartDerivation = np.sqrt(standartDerivation / (len(input)-1))
        return averageLoss, standartDerivation

epochs = 10000
lr = 0.001
hiddenUnits = 2
tp = 'hidden units'


for i in range(5):
    annModel = ArtificialNeuralNetwork(hiddenUnits)
    annModel.train(inputsTrain, outputsTrain, epochs, lr)
    annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
    hiddenUnits = hiddenUnits * 2
print("----------------------------------")

bestUnit = 32
lr = 0.01
tp = 'learning rate'

for i in range(4):
    annModel = ArtificialNeuralNetwork(bestUnit)
    annModel.train(inputsTrain, outputsTrain, epochs, lr)
    annModel.predict(inputsTrain, outputsTrain, tp, float(lr))
    lr = lr / 10

print("----------------------------------")
bestLr = 0.001 #best learning rate according the results
epochs = 10
tp = 'epoch'

#range is normally 6 but it takes too much time for the people who will control it.
for i in range(6):
    annModel = ArtificialNeuralNetwork(bestUnit)
    annModel.train(inputsTrain, outputsTrain, epochs, bestLr)
    annModel.predict(inputsTrain, outputsTrain, tp, float(epochs))
    epochs = epochs * 10

print("----------------------------------")

print("WITH BEST VARIABLES")
#best fit for train data
bestEpoch = 100000 #best epoch
annModel = ArtificialNeuralNetwork(bestUnit)
annModel.train(inputsTrain, outputsTrain, bestEpoch, bestLr)
annModel.predict(inputsTrain, outputsTrain, tp, float(bestEpoch))
annModel.createPlot(inputsTrain, outputsTrain)

averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
print("Average loss for train with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Standart Derivation =", standartDerivation)

#best fit for test data
bestEpoch = 100000 #best epoch
annModel = ArtificialNeuralNetwork(bestUnit)
annModel.train(inputsTest, outputsTest, bestEpoch, bestLr)
annModel.predict(inputsTest, outputsTest, tp, float(bestEpoch))
annModel.createPlot(inputsTest, outputsTest)

averageLoss, standartDerivation = annModel.calculateResults(inputsTest, outputsTest)
print("Average loss for test with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for test with ", bestUnit, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Standart Derivation =", standartDerivation)

print("----------------------------------")

print("WITH BEST LEARNING RATE AND EPOCH, HIDDEN UNITS CHANGES FOR TRAINING DATA")
hiddenUnits = 2
tp = 'hidden units'
for i in range(5):
    annModel = ArtificialNeuralNetwork(hiddenUnits)
    annModel.train(inputsTrain, outputsTrain, bestEpoch, bestLr)
    annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
    annModel.createPlot(inputsTrain, outputsTrain)
    averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
    print("Average loss for train with ", hiddenUnits, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)
    print("Standart Derivation  for train with ", hiddenUnits, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Standart Derivation =", standartDerivation)
    hiddenUnits = hiddenUnits * 2
print("----------------------------------")

print("----------------------------------")

print("WITH BEST LEARNING RATE AND EPOCH, HIDDEN UNITS CHANGES FOR TEST DATA")
hiddenUnits = 2
tp = 'hidden units'
for i in range(5):
    annModel = ArtificialNeuralNetwork(hiddenUnits)
    annModel.train(inputsTest, outputsTest, bestEpoch, bestLr)
    annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
    annModel.createPlot(inputsTest, outputsTest)
    averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
    print("Average loss for train with ", hiddenUnits, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Average loss =", averageLoss)
    print("Standart Derivation  for train with ", hiddenUnits, " unit, ", bestEpoch, " epoch, ", bestLr, " best learning rate, ", "Standart Derivation =", standartDerivation)
    hiddenUnits = hiddenUnits * 2
print("----------------------------------")


print("TESTING LR AFFECT BY NOT CHANGING OTHERS FOR TRAINING DATA")
hiddenUnits = 16
epochs = 10000
lr = 0.01
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.createPlot(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)


hiddenUnits = 16
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.createPlot(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")

print("TESTING LR AFFECT BY NOT CHANGING OTHERS FOR TEST DATA")
hiddenUnits = 16
epochs = 10000
lr = 0.01
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.createPlot(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculateResults(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")

hiddenUnits = 16
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.createPlot(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculateResults(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")


print("TESTING EPOCH AFFECT BY NOT CHANGING OTHERS FOR TRAINING DATA")
hiddenUnits = 16
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.createPlot(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)


hiddenUnits = 16
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTrain, outputsTrain, epochs, lr)
annModel.predict(inputsTrain, outputsTrain, tp, hiddenUnits)
annModel.createPlot(inputsTrain, outputsTrain)
averageLoss, standartDerivation = annModel.calculateResults(inputsTrain, outputsTrain)
print("Average loss for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for train with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")

print("TESTING EPOCH AFFECT BY NOT CHANGING OTHERS FOR TEST DATA")
hiddenUnits = 16
epochs = 100000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.createPlot(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculateResults(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")

hiddenUnits = 16
epochs = 10000
lr = 0.001
annModel = ArtificialNeuralNetwork(hiddenUnits)
annModel.train(inputsTest, outputsTest, epochs, lr)
annModel.predict(inputsTest, outputsTest, tp, hiddenUnits)
annModel.createPlot(inputsTest, outputsTest)
averageLoss, standartDerivation = annModel.calculateResults(inputsTest, outputsTest)
print("Average loss for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Average loss =", averageLoss)
print("Standart Derivation  for test with ", hiddenUnits, " unit, ", epochs, " epoch, ", lr, " best learning rate, ", "Standart Derivation =", standartDerivation)
print("----------------------------------")


