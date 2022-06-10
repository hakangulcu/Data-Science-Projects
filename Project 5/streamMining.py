"""
Author: Hakan Gulcu
Date: 13.05.2022
"""

"""
1) Dataset Generation
skmultiflow.data.HyperplaneGenerator
class skmultiflow.data.HyperplaneGenerator(random_state=None, n_features=10, n_drift_features=2, mag_change=0.0, noise_percentage=0.05, sigma_percentage=0.1)

2)
    2.1) HT
    skmultiflow.trees.HoeffdingTreeClassifier
    class skmultiflow.trees.HoeffdingTreeClassifier(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200, split_criterion='info_gain', 
    split_confidence=1e-07, tie_threshold=0.05, binary_split=False, stop_mem_management=False, 
    remove_poor_atts=False, no_preprune=False, leaf_prediction='nba', nb_threshold=0, nominal_attributes=None)

    2.2) KNN
    skmultiflow.lazy.KNNClassifier
    class skmultiflow.lazy.KNNClassifier(n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean')

    2.3) NB
    skmultiflow.bayes.NaiveBayes
    class skmultiflow.bayes.NaiveBayes(nominal_attributes=None)
"""
# Creating a stream from a csv
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
from skmultiflow.data import HyperplaneGenerator
import numpy as np
import pandas as pandas
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import DynamicWeightedMajorityClassifier

from skmultiflow.evaluation import EvaluatePrequential

# Generate Datasets
generated_dataset = HyperplaneGenerator(n_features=10, n_drift_features=2, noise_percentage=0.1)
generated_dataset_1 = generated_dataset.next_sample(20000)
generated_dataset_1 = np.append(generated_dataset_1[0], np.reshape(generated_dataset_1[1], (20000,1)), axis=1)
generated_dataset_1_final = pandas.DataFrame(generated_dataset_1)
generated_dataset_1_final = generated_dataset_1_final.astype({10: int})
generated_dataset_1_final.to_csv("Hyperplane Dataset 10_2.csv", index=False)

generated_dataset = HyperplaneGenerator(n_features=10, n_drift_features=2, noise_percentage=0.3)
generated_dataset_2 = generated_dataset.next_sample(20000)
generated_dataset_2 = np.append(generated_dataset_2[0], np.reshape(generated_dataset_2[1], (20000,1)), axis=1)
generated_dataset_2_final = pandas.DataFrame(generated_dataset_2)
generated_dataset_2_final = generated_dataset_2_final.astype({10: int})
generated_dataset_2_final.to_csv("Hyperplane Dataset 30_2.csv", index=False)

generated_dataset = HyperplaneGenerator(n_features=10, n_drift_features=5, noise_percentage=0.1)
generated_dataset_3 = generated_dataset.next_sample(20000)
generated_dataset_3 = np.append(generated_dataset_3[0], np.reshape(generated_dataset_3[1], (20000,1)), axis=1)
generated_dataset_3_final = pandas.DataFrame(generated_dataset_3)
generated_dataset_3_final = generated_dataset_3_final.astype({10: int})
generated_dataset_3_final.to_csv("Hyperplane Dataset 10_5.csv", index=False)

generated_dataset = HyperplaneGenerator(n_features=10, n_drift_features=5, noise_percentage=0.3)
generated_dataset_4 = generated_dataset.next_sample(20000)
generated_dataset_4 = np.append(generated_dataset_4[0], np.reshape(generated_dataset_4[1], (20000,1)), axis=1)
generated_dataset_4_final = pandas.DataFrame(generated_dataset_4)
generated_dataset_4_final = generated_dataset_4_final.astype({10: int})
generated_dataset_4_final.to_csv("Hyperplane Dataset 30_5.csv", index=False)


"""
generated_dataset = HyperplaneGenerator(n_features = 10, noise_percentage = 0.1, n_drift_features = 2) # noise= 10%, number of drifting features 2
generated_dataset_1 = generated_dataset.next_sample(20000) #20000 instance
generated_dataset_1= np.append(generated_dataset[0], np.reshape(generated_dataset[1], (20000,1)),axis=1) #reshaping the data and appending it into np array,
converted_generated_dataset_1 = pandas.DataFrame(generated_dataset) #conteining np array with pandas
converted_generated_dataset = converted_generated_dataset.astype({10: int}) #casting type for the last column. it should be an integer
converted_generated_dataset.to_csv("Hyperplane Dataset 10_2.csv") #dataset into csv to create a stream with FileStream
print("Hyperplane Dataset 10_2.csv Created")
"""


hyperplane_default_10_2 = FileStream("Hyperplane Dataset 10_2.csv")
hyperplane_default_10_2.name = "Hyperplane Dataset 10_2"

hyperplane_default_30_2 = FileStream("Hyperplane Dataset 30_2.csv")
hyperplane_default_30_2.name = "Hyperplane Dataset 30_2"

hyperplane_default_10_5 = FileStream("Hyperplane Dataset 10_5.csv")
hyperplane_default_10_5.name = "Hyperplane Dataset 10_5"

hyperplane_default_30_5 = FileStream("Hyperplane Dataset 30_5.csv")
hyperplane_default_30_5.name = "Hyperplane Dataset 30_5"

# Data Stream Classification with Three Separate Online Single Classifiers: HT, KNN, NB

print("Python script constructs and trains 3 online classfiers")

# initializing classfiers
HT = HoeffdingTreeClassifier()
KNN = KNNClassifier()
NB = NaiveBayes()

print("---------------")
evaluator = EvaluatePrequential(metrics=['accuracy', 'running_time'], show_plot=True, max_samples=20000) #setting up the evaluator
evaluator.evaluate(stream = hyperplane_default_10_2, model_names = ["HT", "KNN", "NB"], model = [HT, KNN, NB])
evaluator.evaluate(stream = hyperplane_default_30_2, model_names = ["HT", "KNN", "NB"], model = [HT, KNN, NB])
evaluator.evaluate(stream = hyperplane_default_10_5, model_names = ["HT", "KNN", "NB"], model = [HT, KNN, NB])
evaluator.evaluate(stream = hyperplane_default_30_5, model_names = ["HT", "KNN", "NB"], model = [HT, KNN, NB])
print("---------------")


# In the comparison of classifiers
# try different batch sizes (1, 100 ,1000) and discuss if batch sizes are influential in understanding the performance of the methods (in terms of accuracy and runtime).

def batch_size_comparison(streamName, model_names, model):
    print("---------------")
    print("Batch Size Comparison for", streamName.name)
    print("---------------")
    print("Batch Size: 1")
    evaluator = EvaluatePrequential(metrics=['accuracy', 'running_time'], show_plot=True, max_samples=20000, batch_size = 1) #setting up the evaluator
    evaluator.evaluate(stream = streamName, model_names = model_names, model = model)
    print("---------------")
    print("Batch Size: 100")
    evaluator = EvaluatePrequential(metrics=['accuracy', 'running_time'], show_plot=True, max_samples=20000, batch_size = 100) #setting up the evaluator
    evaluator.evaluate(stream = streamName, model_names = model_names, model = model)
    print("---------------")
    print("Batch Size: 1000")
    evaluator = EvaluatePrequential(metrics=['accuracy', 'running_time'], show_plot=True, max_samples=20000, batch_size = 1000) #setting up the evaluator
    evaluator.evaluate(stream = streamName, model_names = model_names, model = model)
    print("---------------")

batch_size_comparison(hyperplane_default_10_2, ["HT", "KNN", "NB"], [HT, KNN, NB])
batch_size_comparison(hyperplane_default_30_2, ["HT", "KNN", "NB"], [HT, KNN, NB])
batch_size_comparison(hyperplane_default_10_5, ["HT", "KNN", "NB"], [HT, KNN, NB])
batch_size_comparison(hyperplane_default_30_5, ["HT", "KNN", "NB"], [HT, KNN, NB])


dataset_hyperplane_10_2 = pandas.read_csv("Hyperplane Dataset 10_2.csv").values
dataset_hyperplane_30_2 = pandas.read_csv("Hyperplane Dataset 30_2.csv").values
dataset_hyperplane_10_5 = pandas.read_csv("Hyperplane Dataset 10_5.csv").values
dataset_hyperplane_30_5 = pandas.read_csv("Hyperplane Dataset 30_5.csv").values

dataset_hyperplane_10_2_features = dataset_hyperplane_10_2[:,:10]
dataset_hyperplane_10_2_labels = np.array(dataset_hyperplane_10_2[:,10], dtype=int)
train_features_10_2, test_features_10_2, train_labels_10_2, test_labels_10_2 = train_test_split(dataset_hyperplane_10_2_features, dataset_hyperplane_10_2_labels, test_size=0.25)

dataset_hyperplane_30_2_features = dataset_hyperplane_30_2[:,:10]
dataset_hyperplane_30_2_labels = np.array(dataset_hyperplane_30_2[:,10], dtype=int)
train_features_30_2, test_features_30_2, train_labels_30_2, test_labels_30_2 = train_test_split(dataset_hyperplane_30_2_features, dataset_hyperplane_30_2_labels, test_size=0.25)

dataset_hyperplane_10_5_features = dataset_hyperplane_10_5[:,:10]
dataset_hyperplane_10_5_labels = np.array(dataset_hyperplane_10_5[:,10], dtype=int)
train_features_10_5, test_features_10_5, train_labels_10_5, test_labels_10_5 = train_test_split(dataset_hyperplane_10_5_features, dataset_hyperplane_10_5_labels, test_size=0.25)

dataset_hyperplane_30_5_features = dataset_hyperplane_30_5[:,:10]
dataset_hyperplane_30_5_labels = np.array(dataset_hyperplane_30_5[:,10], dtype=int)
train_features_30_5, test_features_30_5, train_labels_30_5, test_labels_30_5 = train_test_split(dataset_hyperplane_30_5_features, dataset_hyperplane_30_5_labels, test_size=0.25)

HT = HoeffdingTreeClassifier()
KNN = KNNClassifier()
NB = NaiveBayes()

def accuracy_calculator(train_X_x, train_Y_y, test_X_x, test_Y_y, name):
    HT = HoeffdingTreeClassifier()
    KNN = KNNClassifier()
    NB = NaiveBayes()

    HT.fit(train_X_x, train_Y_y)
    p1 = HT.predict(test_X_x)

    KNN.fit(train_X_x, train_Y_y)
    p2 = KNN.predict(test_X_x)
    
    NB.fit(train_X_x, train_Y_y)
    p3 = NB.predict(test_X_x)

    print("Batch Hoeffding Tree Accuracy: ",
          (float) (accuracy_score(test_Y_y, p1)), " for dataset ", name)
    print("Batch KNN Accuracy: ", accuracy_score(
        test_Y_y, p2), " for dataset ", name)
    print("Batch Naive Bayes Accuracy: ", accuracy_score(
        test_Y_y, p3), " for dataset ", name)

print("-------------------")
accuracy_calculator(train_features_10_2, train_labels_10_2, test_features_10_2, test_labels_10_2, hyperplane_default_10_2.name)
accuracy_calculator(train_features_30_2, train_labels_30_2, test_features_30_2, test_labels_30_2, hyperplane_default_30_2.name)
accuracy_calculator(train_features_10_5, train_labels_10_5, test_features_10_5, test_labels_10_5, hyperplane_default_10_5.name)
accuracy_calculator(train_features_30_5, train_labels_30_5, test_features_30_5, test_labels_30_5, hyperplane_default_30_5.name)

def voting(train_X_x, train_Y_y, test_X_x, test_Y_y, name):
    HT = HoeffdingTreeClassifier()
    KNN = KNNClassifier()
    NB = NaiveBayes()

    MV = VotingClassifier(estimators=[('HT', HT), ('KNN', KNN), ('NB', NB)], voting='hard', weights=[1, 1, 1])
    MV.fit(train_X_x, train_Y_y)
    p1 = MV.predict(test_X_x)

    print("Voting Accuracy: ", accuracy_score(test_Y_y, p1), " for dataset ", name)

    WMV = VotingClassifier(estimators=[('HT', HT), ('KNN', KNN), ('NB', NB)], voting='hard')
    WMV.fit(train_X_x, train_Y_y)
    p2 = WMV.predict(test_X_x)

    print("Weighted Voting Accuracy: ", accuracy_score(test_Y_y, p2), " for dataset ", name)

# Write a script in Python that constructs and trains the following online classifiers using the four Hyperplane Datasets generated in step 1.
"""
def voting_classifier_accuracy(test_X,p1,p2,p3):
    result = np.array([])
    for i in range(len(test_X)):
        result = np.append(result, statistics.mode([p1[i],p2[i],p3[i]]))
    return accuracy_score(test_X[:,10],result)
"""

def votiing (train_X, train_y, test_X, test_Y, name):

    HT = HoeffdingTreeClassifier()
    KNN = KNNClassifier()
    NB = NaiveBayes()

    HT.fit(train_X, train_y)
    KNN.fit(train_X, train_y)
    NB.fit(train_X, train_y)

    p1 = HT.predict(test_X)
    p2 = KNN.predict(test_X)
    p3 = NB.predict(test_X)

    result = np.array([])
    for i in range(len(test_X)):
        result = np.append(result, statistics.mode([p1[i],p2[i],p3[i]]))
    return result

print("-------------------")
voting(train_features_10_2, train_labels_10_2, test_features_10_2, test_labels_10_2, hyperplane_default_10_2.name)
voting(train_features_30_2, train_labels_30_2, test_features_30_2, test_labels_30_2, hyperplane_default_30_2.name)
voting(train_features_10_5, train_labels_10_5, test_features_10_5, test_labels_10_5, hyperplane_default_10_5.name)
voting(train_features_30_5, train_labels_30_5, test_features_30_5, test_labels_30_5, hyperplane_default_30_5.name)

#Test whether we can improve accuracy
KNN = KNNClassifier()
DWMC = DynamicWeightedMajorityClassifier(KNN)

KNN_online_classifier = KNNClassifier()

evaluator = EvaluatePrequential(metrics=['accuracy', 'running_time'], show_plot=True, max_samples=20000)

evaluator.evaluate(stream=hyperplane_default_10_2, model=[KNN_online_classifier, DWMC], model_names=["KNN", "DWMC"])
evaluator.evaluate(stream=hyperplane_default_30_2, model=[KNN_online_classifier, DWMC], model_names=["KNN", "DWMC"])
evaluator.evaluate(stream=hyperplane_default_10_5, model=[KNN_online_classifier, DWMC], model_names=["KNN", "DWMC"])
evaluator.evaluate(stream=hyperplane_default_30_5, model=[KNN_online_classifier, DWMC], model_names=["KNN", "DWMC"])