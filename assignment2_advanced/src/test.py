import pickle
import glob
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from assignment2_advanced.src.utils.load_fer import load_fer_data

def test_fer_model(img_folder, model="fer_model.pickle", mean_image_file='mean_image.pickle'):
    """
    Given a folder with images, load the images and your best model to predict the facial expressions of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in img_folder.
    """
    if not img_folder.endswith('/'):
        img_folder += '/'

    preds = None

    # load images
    #X_train, y_train, X_val, y_val = load_fer_data(img_folder, 0, 100)
    test_filelist = sorted(glob.glob(img_folder + '*.jpg'))
    print('test_filelist:', test_filelist[:5])
    num_test_images = len(test_filelist)
    X_test = np.ndarray(shape=(num_test_images, 48, 48, 3), dtype=np.float32)
    for i in range(num_test_images):
        if i % 100 == 0:
            print(i, 'of', num_test_images, 'testing images loaded')
        X_test[i] = img_to_array(load_img(test_filelist[i]))

    # subtract mean image
    mean_image = pickle.load(open(mean_image_file, 'rb'))
    X_test -= mean_image

    # make grayscale
    X_test = X_test.mean(axis=3)

    # load model
    final_model = pickle.load(open(model, 'rb'))

    # predict expressions
    scores = final_model.loss(X_test)
    preds = np.argmax(scores, axis=1)

    return preds

if __name__ == '__main__':
    import os
    import pandas as pd

    PATH = os.getcwd()
    print('PATH:', PATH)
    img_folder = PATH + '/datasets/FER2013/Test/'
    predictions = test_fer_model(img_folder)
    print('preds:', predictions[:5])

    labels = pd.read_table('datasets/FER2013/labels_public.txt', delimiter=',').sort_values('img')
    test_df = labels[labels['img'].apply(lambda x: x[:4] == 'Test')].sort_values('img').reset_index()
    print('test_df:', test_df.head(5))
