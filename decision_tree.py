# coding: utf-8

#### CW1 395 Machine Learning

####################################   SetUp    ####################################

import numpy as np
import scipy.io

clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
noisy_data = scipy.io.loadmat("./Data/noisydata_students.mat")

x_clean = clean_data.get("x")
print("n_examples = " + str(x_clean.shape[0]) + " action_units = " + str(x_clean.shape[1]))

y_clean = clean_data.get("y")
emotions = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "sadness", 5: "surprise"}

x_noisy = noisy_data.get("x")
y_noisy = noisy_data.get("y")

####################################   Decision Tree    ####################################


class Tree:
    def __init__(self, rootid):
        self.left = None
        self.right = None
        self.rootid = rootid  # will be attributes/action units
        self.kids = []  # counting the left and right children of the current node
        self.op = None  # label of the attribute that is being tested
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

    def getNodeValue(self):
        return self.rootid

    def setNodeValue(self, value):
        self.rootid = value

    def setKids(self, value):
        self.kids = value

    def setLabel(self, value):
        self.label = value

    def insertRight(self, newNode):
        if self.right == None:
            self.right = Tree(newNode)
        else:
            current_tree = Tree(newNode)
            Tree.right = self.right
            self.right = current_tree

    def insertLeft(self, newNode):
        if self.left == None:
            self.left = Tree(newNode)
        else:
            current_tree = Tree(newNode)
            self.left = current_tree
            tree.left = self.left

def printTree(tree):
    if tree != None:
        printTree(tree.getLeftChild())
        print(tree.getNodeValue())
        printTree(tree.getRightChild())

myTree = Tree(43)
myTree.insertLeft(41)
myTree.insertRight(21)
myTree.insertRight(20)
printTree(myTree)


def subset(x_data, y_data, target):
    # x_data(array[N,A]): the data for which a subset should be created with
    #                     N examples and A attributes
    # y_data(array[N]):   the labeled data for all N examples
    # target(range(1,7)): the target emotion
    # output:             a target vector, which has 0 for all y_data != target and
    #                                                1 for all y_data == target
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
    # p = number of positive examples
    if (p + n) == 0: # to avoid division with 0
        return 0
    elif p == 0:     # to avoid division with 0
        return - n / (p + n) * np.log2(n / (p + n))
    elif n == 0:     # to avoid division with 0
        return -p / (p + n) * np.log2(p / (p + n))
    else:
        return -p / (p + n) * np.log2(p / (p + n)) - n / (p + n) * np.log2(n / (p + n))


def choose_best_decision_attribute(examples, attributes, binary_target):
    max_gain = -1
    p = np.sum(binary_target)
    n = len(binary_target) - p
    I = entropy(p, n)
    max_index = -1
    print("initial entropy = " + str(I) + "\n")
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


choose_best_decision_attribute(x_clean, np.zeros(45), bin_anger)

import collections


def decision_tree_learning(examples, attributes, binary_target):
    # examples (array[N,A])     : examples with N number of examples,
    #                             A number of attributes (Action Units)
    # attributes (array[1,A])   : a vector with all available attributes A
    # binary_target (array[N,1]):
    if np.all(binary_target == binary_target[0], axis=0):
        print("all the same")
        return (binary_target[0])
    elif len(attributes) == 0:
        print("attributes are empty")
        return (round(np.sum(binary_target)))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_target)
        current_attribute = attributes[best_attribute]
        tree = Tree(attributes[best_attribute])
        del (attributes[best_attribute])

        examples_0 = examples[examples[:, best_attribute] == 0]
        examples_0 = np.delete(examples_0, best_attribute, 1)
        binary_target0 = binary_target[examples[:, best_attribute] == 0]
        print("binary_target0" + str(collections.Counter(binary_target0)))
        if len(examples_0) == 0:
            tree.setLabel(0)
        else:
            tree.setNodeValue(current_attribute)
            tree.insertLeft(decision_tree_learning(examples_0, attributes, binary_target0))

        examples_1 = examples[examples[:, best_attribute] == 1]
        examples_1 = np.delete(examples_1, best_attribute, 1)
        binary_target1 = binary_target[examples[:, best_attribute] == 1]

        print("binary_target1" + str(collections.Counter(binary_target1)))

        if len(examples_1):
            tree.setLabel(1)
        else:
            tree.setNodeValue(current_attribute)
            tree.insertRight(decision_tree_learning(examples_1, attributes, binary_target1))
    return tree


action_units = []
for i in range(x_clean.shape[1]):
    action_units.append("AU" + str(i))
dec_tree = decision_tree_learning(x_clean, action_units, bin_anger)

####################################   Tests    ####################################



####################################   K-fold Cross validation    ####################################
