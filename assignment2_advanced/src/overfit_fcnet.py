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
# Parameters
hidden_dims = [50]
update_rule = 'sgd' # ['sgd', 'sgd_momentum']
num_epochs = 20

# Get data and limit to 50 samples (?)
data = get_CIFAR10_data()

# Create net and train
net = FullyConnectedNet(hidden_dims, input_dim=32*32*3, num_classes=10, dropout=0)
slvr = Solver(model, data, update_rule=update_rule, num_epochs=num_epochs)
slvr.train()

# Check accuracy
slvr.check_accuracy(data['X_test'], data['y_test'])

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
