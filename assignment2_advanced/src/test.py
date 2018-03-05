from assignment2_advanced.src.utils.load_fer import load_fer_data

def test_fer_model(img_folder, model="/path/to/model"):
    """
    Given a folder with images, load the images and your best model to predict the facial expressions of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in img_folder.
    """
    preds = None

    # load images
    #X_train, y_train, X_val, y_val = load_fer_data(img_folder, 0, 100)
    test_filelist = sorted(glob.glob(img_folder + '/*.jpg'))
    num_test_images = len(test_filelist)
    X_test = np.ndarray(shape=(num_test_images, 48, 48, 3), dtype=np.float32)
    for i in range(num_test_images):
        if i % 100 == 0:
            print(i, 'of', num_test_images, 'testing images loaded')
        X_test[i] = img_to_array(load_img(test_filelist[i]))
    data = {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
    }

    # load model


    # predict expressions

    return preds

if __name__ == '__main__':
    import os

    PATH = os.getcwd()
    print('PATH:', PATH)
    img_folder = PATH + '/datasets/FER2013'
    test_fer_model(img_folder, model="/path/to/model")
