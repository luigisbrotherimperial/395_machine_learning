import numpy as np
import matplotlib.pyplot as plt
import os

from assignment2_advanced.src.fcnet import FullyConnectedNet
from assignment2_advanced.src.utils.solver import Solver
from assignment2_advanced.src.utils.load_fer import load_fer_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

# settings
num_train_images = 10000 # out of 28708
num_test_images = 1000 # out of 3588

# load images
PATH = os.getcwd()
print('PATH:', PATH)
img_folder = PATH + '/datasets/FER2013'
X_train, y_train, X_val, y_val = load_fer_data(img_folder, num_train_images, num_test_images)
data = {
  'X_train': X_train, 'y_train': y_train,
  'X_val': X_val, 'y_val': y_val,
}

# Create net and solver and train
model = FullyConnectedNet(hidden_dims=[60, 75],
                          input_dim=np.prod(data['X_train'].shape[1:]),
                          num_classes=len(np.unique(data['y_train'])),
                          dropout=0,
                          reg=0.2,
                          weight_scale=1e-2)

solver = Solver(model, data,
                update_rule='sgd_momentum', # ['sgd', 'sgd_momentum']
                optim_config={
                  'learning_rate': 2e-3,
                },
                lr_decay=0.95,
                num_epochs=100,
                batch_size=100,
                print_every=100)
solver.train()

# Plot errors
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
