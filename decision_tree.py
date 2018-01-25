# coding: utf-8

# ### CW1 395 Machine Learning

# ### Setup


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


# ### Decision Tree:

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
            current_tree.right = self.right
            self.right = current_tree

    def insertLeft(self, newNode):
        if self.left == None:
            self.left = Tree(newNode)
        else:
            current_tree = Tree(newNode)
            self.left = current_tree
            current_tree.left = self.left

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

print(np.sum(bin_anger) + np.sum(bin_disgust) + np.sum(bin_fear) + np.sum(bin_happiness) + np.sum(bin_sadness) + np.sum(
    bin_surprise))


def decision_tree_learning(examples, attributes, binary_target):
    # examples (array[N,A])     : examples with N number of examples,
    #                             A number of attributes (Action Units)
    # attributes (array[1,A])   : a vector with all available attributes A
    # binary_target (array[N,1]):
    return


# ### K-folds Cross Validation

