import glob
from PIL import Image
import pandas as pd
import numpy as np

def load_fer_data(path_train = "./Train/*.jpg", path_test = "./Test/*.jpg", path_labels = "./labels_public.txt"):
    train_filelist = glob.glob(path_train)
    test_filelist = glob.glob(path_test)
    labels = pd.read_table(path_labels, delimiter=",")
    X_train = np.array([np.array(Image.open(fname)) for fname in train_filelist])
    y_train = np.array(labels[labels['img'].str.match('Train')]["emotion"])
    X_test = np.array([np.array(Image.open(fname)) for fname in test_filelist])
    y_test = np.array(labels[labels['img'].str.match('Test')]["emotion"])
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape:  " + str(X_test.shape))
    print("y_test shape:  " + str(y_test.shape))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_fer_data()