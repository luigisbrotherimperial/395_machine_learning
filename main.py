# coding: utf-8

#### CW1 395 Machine Learning

import numpy as np
import scipy.io
from decision_tree import *

#######################################         Initialise         #######################################

clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
noisy_data = scipy.io.loadmat("./Data/noisydata_students.mat")

x_clean = clean_data.get("x")
print("n_examples = " + str(x_clean.shape[0]) + " action_units = " + str(x_clean.shape[1]))

y_clean = clean_data.get("y")
emotions = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "sadness", 5: "surprise"}

x_noisy = noisy_data.get("x")
y_noisy = noisy_data.get("y")


#######################################         Define Emotions         #######################################
print("\nDefine Emotions")
bin_anger = subset(x_clean, y_clean, 1)
bin_disgust = subset(x_clean, y_clean, 2)
bin_fear = subset(x_clean, y_clean, 3)
bin_happiness = subset(x_clean, y_clean, 4)
bin_sadness = subset(x_clean, y_clean, 5)
bin_surprise = subset(x_clean, y_clean, 6)

#######################################         Learning         #######################################

print("\nLearning")
dec_tree_anger     = decision_tree_learning(x_clean, range(45), bin_anger)
dec_tree_disgust   = decision_tree_learning(x_clean, range(45), bin_disgust)
dec_tree_fear      = decision_tree_learning(x_clean, range(45), bin_fear)
dec_tree_happiness = decision_tree_learning(x_clean, range(45), bin_happiness)
dec_tree_sadness   = decision_tree_learning(x_clean, range(45), bin_sadness)
dec_tree_surprise   = decision_tree_learning(x_clean, range(45), bin_surprise)

# TODO: Visualization of the trees:

#######################################         Test general setup         #######################################

print("\nTest general setup")
# for perfectly trained trees:
x_test = x_clean[500:1000]
y_test_anger = bin_anger[500:1000]
y_test_disgust = bin_disgust[500:1000]
y_test_fear = bin_fear[500:1000]
y_test_happiness = bin_happiness[500:1000]
y_test_sadness = bin_sadness[500:1000]
y_test_surprise = bin_surprise[500:1000]

print("test accuracy for perfect decision tree (anger)     = " + str(test_accuracy(x_test, y_test_anger, dec_tree_anger)*100)+str("%%"))
print("test accuracy for perfect decision tree (disgust)   = " + str(test_accuracy(x_test, y_test_disgust, dec_tree_disgust)*100)+str("%%"))
print("test accuracy for perfect decision tree (fear)      = " + str(test_accuracy(x_test, y_test_fear, dec_tree_fear)*100)+str("%%"))
print("test accuracy for perfect decision tree (happiness) = " + str(test_accuracy(x_test, y_test_happiness, dec_tree_happiness)*100)+str("%%"))
print("test accuracy for perfect decision tree (sadness)   = " + str(test_accuracy(x_test, y_test_sadness, dec_tree_sadness)*100)+str("%%"))
print("test accuracy for perfect decision tree (surprise)  = " + str(test_accuracy(x_test, y_test_surprise, dec_tree_surprise)*100)+str("%%"))


#######################################         Test all emotions         #######################################

print("\nTest All Emotions")
trees = [dec_tree_anger, dec_tree_disgust, dec_tree_fear, dec_tree_happiness, dec_tree_sadness, dec_tree_surprise]

prediction_all_emotions = []
for i in range(x_clean.shape[0]):
    prediction_all_emotions.append(testTrees(trees, x_clean[i]))
prediction_all_emotions = np.reshape(np.array(prediction_all_emotions), [x_clean.shape[0], 1])

print("test accuracy for perfect decision tree (all emotions) = " + str(round(np.sum(prediction_all_emotions == y_clean)/len(y_clean), 2)*100) + str("%%"))

print("\nK Folds:")
k_fold_cross_validation(10, x_clean, y_clean)

print("\nConfusion matrix")
#test confusion  matrix
confusion_matrix(prediction_all_emotions, y_clean)

#USED TO TEST   
cmat = confusion_matrix(prediction_all_emotions, y_clean)
print(cmat)
recall_rate = recall(cmat)
precision_rate = precision(cmat)
f1_measure(precision_rate, recall_rate)
classification_rate(cmat)
