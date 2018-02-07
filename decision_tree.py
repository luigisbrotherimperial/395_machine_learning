import numpy as np
from random import randint, choice
#######################################         Decision Tree         #######################################
class Tree:
    def __init__(self, op):
        self.op = op  # label of the attribute that is being tested
        self.kids = [0, 0]  # counting the left and right children of the current node
        self.label = None  # label for leaf nodes

    def getLeftChild(self):
        return self.kids[0]

    def getRightChild(self):
        return self.kids[1]

    def getOp(self):
        return self.op

    def getLabel(self):
        return self.label

    def setOp(self, value):
        self.op = value

    def setLabel(self, value):
        self.label = value



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


def entropy(p, n):
    if (p + n) == 0:
        return 0
    elif p == 0:
        return - n / (p + n) * np.log2(n / (p + n))
    elif n == 0:
        return -p / (p + n) * np.log2(p / (p + n))
    else:
        return -p / (p + n) * np.log2(p / (p + n)) - n / (p + n) * np.log2(n / (p + n))
        

#CHANGED! new function added         
def majority_value(binary_target):
    
    return int(round(np.sum(binary_target) / len(binary_target)))


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
        
        #CHANGED - leaf node empty op
        tree = Tree('')
        tree.setLabel(int(binary_target[0]))
        return tree
    elif len(attributes) == 0:
        
        #CHANGED
        mode = majority_value(binary_target)
        
        #CHANGED - leaf node empty op
        tree = Tree('')
        tree.setLabel(mode)
        
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


        #CHANGED! 
        if (len(examples_0) == 0) or (len(examples_1) == 0):
            
            #CHANGED! 
            mode = majority_value(binary_target)
            
            #CHANGED - leaf node empty op
            tree = Tree('')
            tree.setLabel(mode)
            
        else:
            # TODO: count Kids!
            leftTree = decision_tree_learning(examples_0, attributes, binary_target0)
            tree.kids[0]=leftTree

            rightTree = decision_tree_learning(examples_1, attributes, binary_target1)
            tree.kids[1]=rightTree
            
    return tree

#######################################         Prediction         #######################################

def prediction(decision_tree, x_data):
    while decision_tree.getLabel() == None:
        op = decision_tree.getOp()
        if (x_data[op] == 0):
            decision_tree = decision_tree.getLeftChild()
        else:
            decision_tree = decision_tree.getRightChild()
    return decision_tree.getLabel()


def test_accuracy(x_test, y_test, decision_tree):
    predictions = []
    for i in range(x_test.shape[0]):
        predictions.append(prediction(decision_tree, x_test[i]))
    predictions = np.array(predictions)
    return np.sum(predictions == y_test) / len(y_test)


def testTrees(T,x_data):
    # Choose first emotion found
    for i in range(6):
        if prediction(T[i], x_data) == 1:
            return i+1
    return randint(1,6)

def testTrees2(T,x_data):
    # Choose randomly from predicted emotions
    predicted_emotions = []
    for i in range(6):
        if prediction(T[i], x_data) == 1:
            predicted_emotions.append(i+1)
    if len(predicted_emotions) == 0:
        return randint(1, 6)
    else:
        return choice(predicted_emotions)

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


#######################################         confusion matrix         #######################################

def confusion_matrix(predicted, actual):
    
    #predefine the confusion matrix
    cmat=np.zeros((6,6))
    print(cmat.shape)
      
    #increment the matrix
    for i in range(len(predicted)):
        cmat[actual[i]-1, predicted[i]-1]+=1  
      
    #return the confusion matrix
    return cmat 

<<<<<<< Updated upstream
<<<<<<< Updated upstream
#######################################         recall        #######################################

def recall(conf_mnatrix):
    
    #initialize the recall rate array
    rate = np.zeros((6))
    
    #compute the recall rate for each class
    for i in range(6):
        
        rate[i] = conf_mnatrix[i, i]*100/sum(conf_mnatrix[i, :])
     
    #return the recall rate 
    #print(rate)       
    return rate
    
    
#######################################         precision        #######################################

def precision(conf_mnatrix):
     
    #initialize the precision rate array
    rate = np.zeros((6))
    
    #compute the precision rate for each class
    for i in range(6):
        
        rate[i] = conf_mnatrix[i, i]*100/sum(conf_mnatrix[:, i])
     
    #return the precision rate  
    #print(rate)      
    return rate   
 
 
#USED TO TEST       
#cmat = confusion_matrix(prediction_all_emotions, y_clean)
#print(cmat)
#recall(cmat)
#precision(cmat)
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
