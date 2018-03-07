395 Machine Learning: Assignment 2
=====================================


### Required packages
* pandas
* keras v 2.1.4
* tensorflow 1.6.0
* Pillow
* bayesian-optimization
* h5py 
* scikit-image

### Running test.py for CNN
[test.py](../assignment2_advanced/src/test.py) is setup as a unittest file. Please, if that is not intended, 
 take the file [cnn_test.py](../assignment2_advanced/src/cnn_test.py) instead.

For running the model with your test data, please either delete the [Test](../assignment2_advanced/data/Test)
and [labels](../assignment2_advanced/data/labels) folders and put new Test and labels folders with all your images 
exactly in the same place or specify the paths to your data in test.py accordingly. 

The functions in test.py and cnn_test.py also do the appropriate preprocessing for testing. 
