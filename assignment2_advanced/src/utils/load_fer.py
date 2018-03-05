import glob
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def load_fer_data(img_folder, num_train=28708, num_test=3588):
    path_train = img_folder + "/Train/*.jpg"
    path_test = img_folder + "/Test/*.jpg"
    path_labels = img_folder + "/labels_public.txt"

    train_filelist = glob.glob(path_train)[:num_train]
    test_filelist = glob.glob(path_test)[:num_test]
    labels = pd.read_table(path_labels, delimiter=",")

    print('Loading X_train...')
    #X_train = np.array([np.array(Image.open(fname)) for fname in train_filelist])
    X_train = np.ndarray(shape=(num_train, 48, 48, 3), dtype=np.float32)
    for i in range(num_train):
        if i % 100 == 0:
            print(i, 'of', num_train, 'training images loaded')
        X_train[i] = img_to_array(load_img(train_filelist[i]))

    print('Loading y_train...')
    y_train = np.array(labels[labels['img'].str.match('Train')]["emotion"][:num_train])

    print('Loading X_test...')
    #X_test = np.array([np.array(Image.open(fname)) for fname in test_filelist])
    X_test = np.ndarray(shape=(num_test, 48, 48, 3), dtype=np.float32)
    for i in range(num_test):
        if i % 100 == 0:
            print(i, 'of', num_test, 'testing images loaded')
        X_test[i] = img_to_array(load_img(test_filelist[i]))

    print('Loading y_test...')
    y_test = np.array(labels[labels['img'].str.match('Test')]["emotion"][:num_test])

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape:  " + str(X_test.shape))
    print("y_test shape:  " + str(y_test.shape))
    return X_train, y_train, X_test, y_test
