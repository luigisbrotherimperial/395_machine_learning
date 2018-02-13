# 395_machine_learning
## Required packages
This program is written for Python 3+

Visualisation requires pydot and graphviz
```
sudo apt install python-pydot python-pydot-ng graphviz
```
Other packages: NumPy, SciPy, Pickle

## What to run
Run test.py to:
 * load the data into numpy arrays
 * import all decision trees from pickle files
 * get predictions using the testTrees function
 * print the evaluation metrics

Run main.py to:
 * load the data into numpy arrays
 * train decision trees on all 6 emotions
 * test the accuracy of the decision trees on the clean and noisy data
 * get results for 10-folds cross validation
 * visualize the trees (the results will be saved in /results)
 * save the trees as .p files (the results will be saved in /results)
