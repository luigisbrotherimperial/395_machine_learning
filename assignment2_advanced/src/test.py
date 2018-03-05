import os
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
    X_train, y_train, X_val, y_val = load_fer_data(img_folder, 1000, 100)
    data = {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
    }

    # load model
    model = FullyConnectedNet(hidden_dims=[60, 75],
                              input_dim=32*32*3,
                              num_classes=10,
                              dropout=0,
                              reg=0.2,
                              weight_scale=1e-2)

    solver = Solver(model, data,
                    update_rule='sgd_momentum', # ['sgd', 'sgd_momentum']
                    optim_config={
                      'learning_rate': 2e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=40,
                    batch_size=100,
                    print_every=100)
    solver.train()

    # predict expressions

    return preds

if __name__ == '__main__':
    PATH = os.getcwd()
    print('PATH:', PATH)
    img_folder = PATH + '/datasets/FER2013'
    test_fer_model(img_folder, model="/path/to/model")
