import matplotlib.pyplot as plot
import pandas as pd
import numpy
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('falldetection_dataset.csv', header=None) #reading dataset with pandas

labels = data[1] #taking labels of dataset 
features = data.drop(labels=[0,1], axis = 1) #taking features of dataset 
x_trainData = features.values #x train data
y_trainData = labels.values #y train data

#sklearn library has a function called decomposition to achieve PCA 
pcaResult = PCA(n_components = 2) #pca with 2 component
pcaResult.fit(x_trainData)
transformedPCA = pcaResult.transform(x_trainData) 
pcaComponents = pcaResult.components_
pcaEigenVectors = pcaResult.explained_variance_ratio_ #pca eigen vectors with this 

#actions and results

plot.scatter(transformedPCA[y_trainData == 'F', 0], transformedPCA[y_trainData == 'F', 1], label="Fall Action")
plot.scatter(transformedPCA[y_trainData == 'NF', 0], transformedPCA[y_trainData == 'NF', 1], label="Non-Fall Action")
plot.legend()
plot.title("PCA with 2 Components")
plot.savefig('pca2component.png')
plot.show()

for i in range(0,2):
   print("Principal component", i , "variance =", pcaEigenVectors[i]) 
print("Top two principal component variance =", sum(pcaEigenVectors[:i]))

#k-means clustering with outliers
for cluster in range(2, 11):
    kmeans = KMeans(n_clusters = cluster) #applying K-menas clustering with sklearn library.
    kmeans.fit(transformedPCA) #fitting pca
    prediction = kmeans.labels_ #best labels
    for j in range(cluster): #nested loop because we need to scatter for each cluster of data. seperate plots migh be bad.
        plot.scatter(transformedPCA[prediction == j, 0], transformedPCA[prediction == j, 1], label = j) 
    plot.title(str(cluster) + "-Means Clustering with Outliers")
    plot.legend()
    plot.savefig("kmeans" + str(cluster) +".png")
    plot.show()

#Taking the clusters obtained when N=2, check the degree of percentage overlap/consistency between the cluster mem-berships and the action labels originally provided.
kmeans = KMeans(n_clusters = 2) #applying K-menas clustering with sklearn library.
kmeans.fit(transformedPCA) #fitting pca
prediction = kmeans.labels_ #best labels

p_label = [1 if prediction_label == "NF" else 0 for prediction_label in y_trainData] 
x = accuracy_score(p_label, prediction) #metrics.accuracy_score may be used to find accuracy
y = accuracy_score(p_label, 1 - prediction)
#removing outliers
if (x >= y): #finding max accuracy
    print("Accuracy of 2-means = ", x)
else:
    print("Accuracy of 2-means = ", y)

outlier1 = numpy.argwhere(transformedPCA[:, 0] == max(transformedPCA[:, 0]))[0, 0] #finding the first outlier
newTransformedPCA_X = numpy.delete(transformedPCA, outlier1, axis = 0) #deleting the first outlier from projected X train data
newTransformedPCA_Y = numpy.delete(y_trainData, outlier1) #deleting the first outlier from Y train data

outlier2 = numpy.argwhere(newTransformedPCA_X[:, 1] == max(newTransformedPCA_X[:, 1]))[0, 0] #finding the second outlier
newTransformedPCA_X = numpy.delete(newTransformedPCA_X, outlier2, axis = 0) #deleting the second outlier from new projected X train data
newTransformedPCA_Y = numpy.delete(newTransformedPCA_Y, outlier2) #deleting the second outlier from new Y train data

plot.scatter(newTransformedPCA_X[newTransformedPCA_Y == 'F', 0], newTransformedPCA_X[newTransformedPCA_Y == 'F', 1], label="Fall Action") #fall action scatters
plot.scatter(newTransformedPCA_X[newTransformedPCA_Y == 'NF', 0], newTransformedPCA_X[newTransformedPCA_Y == 'NF', 1], label="Non-Fall Action") #non-fall action scatters
plot.legend()
plot.title("PCA with 2 Components Without Outliers")
plot.savefig('pca2componentWithoutOutliers.png')
plot.show()

p_label = [1 if prediction_label == "NF" else 0 for prediction_label in y_trainData] 
x = accuracy_score(p_label, prediction) #metrics.accuracy_score may be used to find accuracy
y = accuracy_score(p_label, 1 - prediction)
#removing outliers
if (x >= y): #finding max accuracy
    print("Accuracy of 2-means = ", x)
else:
    print("Accuracy of 2-means = ", y)
#k-means clustering without outliers
for cluster in range(2, 11):
    kmeans = KMeans(n_clusters = cluster) #applying K-menas clustering with sklearn library.
    kmeans.fit(newTransformedPCA_X) #fitting pca
    prediction = kmeans.labels_ #best labels
    for j in range(cluster): #nested loop because we need to scatter for each cluster of data. seperate plots migh be bad.
        plot.scatter(newTransformedPCA_X[prediction == j, 0], newTransformedPCA_X[prediction == j, 1], label =j) 
    plot.title(str(cluster) + "-Means Clustering without Outliers")
    plot.legend()

    plot.savefig("kmeanswithoutoutlier" + str(cluster) +".png")
    plot.show()


################################################## PART B #########################

data = pd.read_csv('falldetection_dataset.csv', header=None) #reading dataset with pandas

labels = data[1] #taking labels of dataset 
features = data.drop(labels=[0,1], axis = 1) #taking features of dataset 
x_trainData = features.values #x train data
y_trainData = labels.values #y train data

#sklearn.model_selection module has train_test_split function to split data easily
x_trainData, x_testData, y_trainData, y_testData = train_test_split(x_trainData, y_trainData, train_size=0.7) # 70% of dataset is train 
x_ValidationData, x_testData, y_ValidationData, y_testData = train_test_split(x_testData, y_testData, test_size=0.5) #there are 30% left
                                                                                                        #half of it is test, other is validation
                                                                                                        #train=396, validation=85, test=85 labels.
"""
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
                        class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""          
c = 1
kernelType = 'linear'
degreeSVM = 5
max_iteration = 10000
gammaV = 'auto'

#firstly, we need to decide kernel type because this affects more than others according to my calculations.
kernelType = 'linear'
print("--------------------------")
print("kernelType = linear")
print("--------------------------")
for i in range(4):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)

kernelType = 'rbf'
print("--------------------------")
print("kernelType = rbf")
print("--------------------------")
for i in range(4):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)

kernelType = 'poly'
print("--------------------------")
print("kernelType = poly")
print("--------------------------")
for i in range(4):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration, gamma=gammaV) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)

kernelType = 'sigmoid'
print("--------------------------")
print("kernelType = sigmoid")
print("--------------------------")
for i in range(4):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)

kernelType = 'linear' #chosen as best
#now we need to check best C
c = 10
print("--------------------------")
print("C changing")
print("--------------------------")
for i in range(5):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    c = c / 10

best_C = 0.001 #chosen as best
c = best_C
#now we need to check best degree
degreeSVM = 5 
print("--------------------------")
print("degree changing")
print("--------------------------")
for i in range(degreeSVM+1):
    svmModel = SVC(C = c, kernel = kernelType, degree = degreeSVM, max_iter= max_iteration) 
    svmModel.fit(x_trainData, y_trainData)
    prediction = svmModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters C =", c, ", kernel =", kernelType, ", degree =", degreeSVM, ", max_iter =", max_iteration)
    degreeSVM -= 1

best_degree = 3
degreeSVM = 3

print("--------------------------")
print("MLP")
print("--------------------------")

################## MLP #################

"""
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, 
                batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
"""

MLP_learning_rate_init = 1 / 10
MLP_max_iter = 10000
MLP_alpha = 1 / 10 #controls amount of regularization to apply network weights
MLP_activation = 'relu'

print("--------------------------")
print("relu")
print("--------------------------")
#first we need to find activation 
for i in range(1):
    mlpModel = MLPClassifier(learning_rate_init = MLP_learning_rate_init, max_iter= MLP_max_iter, alpha = MLP_alpha, activation = MLP_activation)
    mlpModel.fit(x_trainData, y_trainData)
    prediction = mlpModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)

print("--------------------------")
print("logistic")
print("--------------------------")   
MLP_activation = 'logistic'
for i in range(1):
    mlpModel = MLPClassifier(learning_rate_init = MLP_learning_rate_init, max_iter= MLP_max_iter, alpha = MLP_alpha, activation = MLP_activation)
    mlpModel.fit(x_trainData, y_trainData)
    prediction = mlpModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)

print("--------------------------")
print("tanh")
print("--------------------------")
MLP_activation = 'tanh'
for i in range(1):
    mlpModel = MLPClassifier(learning_rate_init = MLP_learning_rate_init, max_iter= MLP_max_iter, alpha = MLP_alpha, activation = MLP_activation)
    mlpModel.fit(x_trainData, y_trainData)
    prediction = mlpModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)

MLP_activation = 'relu' #chosen as best

print("--------------------------")
print("choosing alpha")
print("--------------------------")
for i in range(4):
    mlpModel = MLPClassifier(learning_rate_init = MLP_learning_rate_init, max_iter= MLP_max_iter, alpha = MLP_alpha, activation = MLP_activation)
    mlpModel.fit(x_trainData, y_trainData)
    prediction = mlpModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    MLP_alpha = MLP_alpha / 10

MLP_alpha = 1 / 10000

print("--------------------------")
print("choosing learning rate")
print("--------------------------")
for i in range(4):
    mlpModel = MLPClassifier(learning_rate_init = MLP_learning_rate_init, max_iter= MLP_max_iter, alpha = MLP_alpha, activation = MLP_activation)
    mlpModel.fit(x_trainData, y_trainData)
    prediction = mlpModel.predict(x_ValidationData)
    validationAccuracy = accuracy_score(y_ValidationData, prediction)
    testAccuracy = accuracy_score(y_testData, prediction)
    print("Test Accuracy =", str(testAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    print("Validation Accuracy =", str(validationAccuracy*100) + "% ", "for the hyperparameters learning_rate_init =", MLP_learning_rate_init, 
            ", max_iter =", MLP_max_iter, ", alpha =", MLP_alpha, ", activation =", MLP_activation)
    MLP_learning_rate_init = MLP_learning_rate_init / 10

