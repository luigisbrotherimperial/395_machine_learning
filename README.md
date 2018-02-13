# 395_machine_learning
## Required packages
Visualisation requires pydot and graphviz
```
sudo apt install python-pydot python-pydot-ng graphviz
```
Other packages: NumPy, SciPy, Pickle

## Run program
Run main.py to:
 * load the data into numpy arrays
 * train decision trees on all 6 emotions
 * test the accuracy of the decision trees on the clean and noisy data
 * get results for 10-folds cross validation
 * visualize the trees (the results will be saved in /results)
 * save the trees as .p files (the results will be saved in /results)