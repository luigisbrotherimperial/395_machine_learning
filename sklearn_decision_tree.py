####################################   SetUp    ####################################

import numpy as np
import scipy.io
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus


clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
noisy_data = scipy.io.loadmat("./Data/noisydata_students.mat")

x_clean = clean_data.get("x")
print("n_examples = " + str(x_clean.shape[0]) + " action_units = " + str(x_clean.shape[1]))

y_clean = clean_data.get("y")
emotions = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "sadness", 5: "surprise"}

x_noisy = noisy_data.get("x")
y_noisy = noisy_data.get("y")

####################################   Decision Tree    ####################################
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


# train trees:
dtree_anger     = DecisionTreeClassifier()
dtree_disgust   = DecisionTreeClassifier()
dtree_fear      = DecisionTreeClassifier()
dtree_happiness = DecisionTreeClassifier()
dtree_sadness   = DecisionTreeClassifier()
dtree_surprise  = DecisionTreeClassifier()

dtree_anger.fit(x_clean,bin_anger)
dtree_disgust.fit(x_clean,bin_disgust)
dtree_fear.fit(x_clean,bin_fear)
dtree_happiness.fit(x_clean,bin_happiness)
dtree_sadness.fit(x_clean,bin_sadness)
dtree_surprise.fit(x_clean,bin_surprise)

# export visualization
dot_data_anger = StringIO()
dot_data_disgust = StringIO()
dot_data_fear = StringIO()
dot_data_happiness = StringIO()
dot_data_sadness = StringIO()
dot_data_surprise = StringIO()


export_graphviz(dtree_anger, out_file=dot_data_anger,
                filled=True, rounded=True,
                special_characters=True)
export_graphviz(dtree_disgust, out_file=dot_data_disgust,
                filled=True, rounded=True,
                special_characters=True)
export_graphviz(dtree_fear, out_file=dot_data_fear,
                filled=True, rounded=True,
                special_characters=True)
export_graphviz(dtree_happiness, out_file=dot_data_happiness,
                filled=True, rounded=True,
                special_characters=True)
export_graphviz(dtree_sadness, out_file=dot_data_sadness,
                filled=True, rounded=True,
                special_characters=True)
export_graphviz(dtree_surprise, out_file=dot_data_surprise,
                filled=True, rounded=True,
                special_characters=True)

graph_anger = pydotplus.graph_from_dot_data(dot_data_anger.getvalue())
graph_anger.write_png('results/decision_tree_anger.png')

graph_disgust = pydotplus.graph_from_dot_data(dot_data_disgust.getvalue())
graph_disgust.write_png('results/decision_tree_disgust.png')

graph_fear = pydotplus.graph_from_dot_data(dot_data_fear.getvalue())
graph_fear.write_png('results/decision_tree_fear.png')

graph_happiness = pydotplus.graph_from_dot_data(dot_data_happiness.getvalue())
graph_happiness.write_png('results/decision_tree_happiness.png')

graph_sadness = pydotplus.graph_from_dot_data(dot_data_sadness.getvalue())
graph_sadness.write_png('results/decision_tree_sadness.png')

graph_surprise = pydotplus.graph_from_dot_data(dot_data_surprise.getvalue())
graph_surprise.write_png('results/decision_tree_surprise.png')