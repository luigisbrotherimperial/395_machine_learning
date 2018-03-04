import numpy as np

from assignment2_advanced.src.fcnet import FullyConnectedNet
from assignment2_advanced.src.utils.solver import Solver
from assignment2_advanced.src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

# Get data and limit to 50 samples
data = get_CIFAR10_data(50, 0, 0, True)
data['X_test'] = data['X_train']
data['y_test'] = data['y_train']
data['X_val'] = data['X_train']
data['y_val'] = data['y_train']

# Create net and solver and train
model = FullyConnectedNet(hidden_dims=[50],
                          input_dim=32*32*3,
                          num_classes=10,
                          dropout=0)

solver = Solver(model, data,
                update_rule='sgd', # ['sgd', 'sgd_momentum']
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=20,
                batch_size=100,
                print_every=100)
solver.train()

# Check accuracy on training data
acc = solver.check_accuracy(data['X_train'], data['y_train'])

print('Test set accuracy:', acc)

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
