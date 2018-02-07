# coding: utf-8

#### CW1 395 Machine Learning

#######################################         Setup         #######################################


import numpy as np
import scipy.io
from random import randint

clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
noisy_data = scipy.io.loadmat("./Data/noisydata_students.mat")

x_clean = clean_data.get("x")
print("n_examples = " + str(x_clean.shape[0]) + " action_units = " + str(x_clean.shape[1]))

y_clean = clean_data.get("y")
emotions = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "sadness", 5: "surprise"}

x_noisy = noisy_data.get("x")
y_noisy = noisy_data.get("y")

#######################################         Decision Tree         #######################################


class Tree:
    def __init__(self, op):
        self.left = None
        self.right = None
        self.op = op  # label of the attribute that is being tested
        self.kids = [0, 0]  # counting the left and right children of the current node
        self.label = None  # label for leaf nodes

    def getLeftChild(self):
        return self.left

    def getRightChild(self):
        return self.right

    def getKids(self):
        return self.kids

    def getOp(self):
        return self.op

    def getLabel(self):
        return self.label

    def setOp(self, value):
        self.op = value

    def setLabel(self, value):
        self.label = value

    def setKids(self, value):
        self.kids = value

    def insertRight(self, newNode):
        if self.right == None:
            self.right = newNode
            self.kids[1] = self.kids[1] + 1
        else:
            newNode.right = self.right
            self.right = newNode
            self.kids[1] = self.kids[1] + 1

    def insertLeft(self, newNode):
        if self.left == None:
            self.left = newNode
            self.kids[0] = self.kids[0] + 1
        else:
            newNode.left = self.left
            self.left = newNode
            self.kids[0] = self.kids[0] + 1


def printTree(tree):
    if tree != None:
        print(tree.getOp())
        print()
        printTree(tree.getLeftChild())
        printTree(tree.getRightChild())


def subset(x_data, y_data, target):
    # x_data(array[N,A]): the data for which a subset should be created with
    #                     N examples and A attributes
    # y_data(array[N]):   the labeled data for all N examples
    # target(range(1,7)): the target emotion
    bool_array = y_data == target
    binary_target = np.zeros(x_data.shape[0])
    binary_target[bool_array[:, 0]] = 1
    return binary_target


bin_anger = subset(x_clean, y_clean, 1)
bin_disgust = subset(x_clean, y_clean, 2)
bin_fear = subset(x_clean, y_clean, 3)
bin_happiness = subset(x_clean, y_clean, 4)
bin_sadness = subset(x_clean, y_clean, 5)
bin_surprise = subset(x_clean, y_clean, 6)


def entropy(p, n):
    if (p + n) == 0:
        return 0
    elif p == 0:
        return - n / (p + n) * np.log2(n / (p + n))
    elif n == 0:
        return -p / (p + n) * np.log2(p / (p + n))
    else:
        return -p / (p + n) * np.log2(p / (p + n)) - n / (p + n) * np.log2(n / (p + n))


def choose_best_decision_attribute(examples, attributes, binary_target):
    max_gain = -1
    p = np.sum(binary_target)
    n = len(binary_target) - p
    I = entropy(p, n)
    max_index = -1
    for i in range(len(attributes)):
        # the number of positive examples for the subset of the training data
        # for which the attribute has the value 0
        p0 = np.sum(binary_target[examples[:, i] == 0] == 1)
        # the number of negative examples for the subset of the training data
        # for which the attribute has the value 0
        n0 = np.sum(binary_target[examples[:, i] == 0] == 0)
        # the number of positive examples for the subset of the training data
        # for which the attribute has the value 1
        p1 = np.sum(binary_target[examples[:, i] == 1] == 1)
        # the number of negative examples for the subset of the training data
        # for which the attribute has the value 1
        n1 = np.sum(binary_target[examples[:, i] == 1] == 0)
        current_remainder = (p0 + n0) / (p + n) * entropy(p0, n0) + (p1 + n1) / (p + n) * entropy(p1, n1)
        current_gain = I - current_remainder
        if current_gain > max_gain:
            max_gain = current_gain
            max_index = i
    return max_index


def decision_tree_learning(examples, attributes, binary_target):
    # examples (array[N,A])     : examples with N number of examples,
    #                             A number of attributes (Action Units)
    # attributes (array[1,A])   : a vector with all available attributes A
    # binary_target (array[N,1]):
    if np.all(binary_target == binary_target[0], axis=0):
        tree = Tree(int(binary_target[0]))
        tree.setLabel(int(binary_target[0]))
        return tree
    elif len(attributes) == 0:
        tree = Tree(int(round(np.sum(binary_target) / len(binary_target))))
        tree.setLabel(int(round(np.sum(binary_target) / len(binary_target))))
        return tree
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_target)
        op = attributes[best_attribute]
        tree = Tree(op)
        attributes = np.delete(attributes, best_attribute)

        examples_0 = examples[examples[:, best_attribute] == 0]
        examples_0 = np.delete(examples_0, best_attribute, 1)
        binary_target0 = binary_target[examples[:, best_attribute] == 0]

        examples_1 = examples[examples[:, best_attribute] == 1]
        examples_1 = np.delete(examples_1, best_attribute, 1)
        binary_target1 = binary_target[examples[:, best_attribute] == 1]

        if len(examples_0) == 0:
            tree.setLabel(0)
        else:
            # TODO: count Kids!
            leftTree = decision_tree_learning(examples_0, attributes, binary_target0)
            tree.setKids([leftTree.getKids()[0]+1, leftTree.getKids()[1]])
            tree.insertLeft(leftTree)

        if len(examples_1) == 0:
            tree.setLabel(1)
        else:
            rightTree = decision_tree_learning(examples_1, attributes, binary_target1)
            tree.setKids([rightTree.getKids()[0], rightTree.getKids()[1]+1])
            tree.insertRight(rightTree)
    return tree




#######################################         Learning         #######################################


dec_tree_anger     = decision_tree_learning(x_clean, range(45), bin_anger)
dec_tree_disgust   = decision_tree_learning(x_clean, range(45), bin_disgust)
dec_tree_fear      = decision_tree_learning(x_clean, range(45), bin_fear)
dec_tree_happiness = decision_tree_learning(x_clean, range(45), bin_happiness)
dec_tree_sadness   = decision_tree_learning(x_clean, range(45), bin_sadness)
dec_tree_surprise   = decision_tree_learning(x_clean, range(45), bin_surprise)

# TODO: Visualization of the trees:

#######################################         Prediction         #######################################

def prediction(decision_tree, x_data):
    while decision_tree.getLabel() == None:
        op = decision_tree.getOp()
        if (x_data[op] == 0):
            decision_tree = decision_tree.getLeftChild()
        else:
            decision_tree = decision_tree.getRightChild()
    return decision_tree.getLabel()


#######################################         Test general setup         #######################################

# for perfectly trained trees:
x_test = x_clean[500:1000]
y_test_anger = bin_anger[500:1000]
y_test_disgust = bin_disgust[500:1000]
y_test_fear = bin_fear[500:1000]
y_test_happiness = bin_happiness[500:1000]
y_test_sadness = bin_sadness[500:1000]
y_test_surprise = bin_surprise[500:1000]


def test_accuracy(x_test, y_test, decision_tree):
    predictions = []
    for i in range(x_test.shape[0]):
        predictions.append(prediction(decision_tree, x_test[i]))
    predictions = np.array(predictions)
    return np.sum(predictions == y_test) / len(y_test)

print("test accuracy for perfect decision tree (anger)     = " + str(test_accuracy(x_test, y_test_anger, dec_tree_anger)*100)+str("%%"))
print("test accuracy for perfect decision tree (disgust)   = " + str(test_accuracy(x_test, y_test_disgust, dec_tree_disgust)*100)+str("%%"))
print("test accuracy for perfect decision tree (fear)      = " + str(test_accuracy(x_test, y_test_fear, dec_tree_fear)*100)+str("%%"))
print("test accuracy for perfect decision tree (happiness) = " + str(test_accuracy(x_test, y_test_happiness, dec_tree_happiness)*100)+str("%%"))
print("test accuracy for perfect decision tree (sadness)   = " + str(test_accuracy(x_test, y_test_sadness, dec_tree_sadness)*100)+str("%%"))
print("test accuracy for perfect decision tree (surprise)  = " + str(test_accuracy(x_test, y_test_surprise, dec_tree_surprise)*100)+str("%%"))


#######################################         Test all emotions         #######################################

trees = [dec_tree_anger, dec_tree_disgust, dec_tree_fear, dec_tree_happiness, dec_tree_sadness, dec_tree_surprise]

def testTrees(T,x_data):
    for i in range(6):
        if prediction(T[i], x_data)!=0:
            return i+1
    return randint(1,6)

# TODO: 2nd implementation of testTrees

prediction_all_emotions = []
for i in range(x_clean.shape[0]):
    prediction_all_emotions.append(testTrees(trees, x_clean[i]))
prediction_all_emotions = np.reshape(np.array(prediction_all_emotions), [x_clean.shape[0], 1])
print("test accuracy for perfect decision tree (all emotions) = " + str(round(np.sum(prediction_all_emotions == y_clean)/len(y_clean), 2)*100) + str("%%"))


#######################################         k-folds cross validation         #######################################

def k_fold_cross_validation(k, x_data, y_data):
    equal_parts = int(np.floor(x_data.shape[0]/k))
    trees = []
    for j in range(1,7):
        bin_emotion = subset(x_data, y_data, j)
        for i in range(k):
            # x_test  = x_data[i*equal_parts:equal_parts*(i+1)]
            # y_test  = bin_emotion[i*equal_parts:equal_parts*(i+1)]
            x_train = np.append     (x_data[0:i*equal_parts],      x_data[(i+1)*equal_parts:], axis = 0)
            y_train = np.append(bin_emotion[0:i*equal_parts], bin_emotion[(i+1)*equal_parts:])
            trees.append(decision_tree_learning(x_train, range(45), y_train))
    return trees

# this would return 60 trained trees. the first 10 trees are trained on anger, the second 10 trees on disgust, ...
# k_trees = k_fold_cross_validation(10, x_clean, y_clean)
# TODO: use either for confusion matrix or average values (you can build it directly into k_fold_cross_validation)

