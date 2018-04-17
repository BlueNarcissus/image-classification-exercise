from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
   print(('%s: ' % k, v.shape))

""" Two layer network test"""
""" train model with solver """
""" Fully-Connected Network test """

""" SGD+Momentum """
""" RMSProp & Adam """

""" Best model """
best_model = None
best_solver = None
best_val = -1
counter = 1

reg = 0
lr = 1e-3
learning_rates = [5e-4, 1e-3, 2e-3]
regularizations = [0.0, 1e-2]

for lr in learning_rates:
    for reg in regularizations:
        model = FullyConnectedNet([100, 100, 100, 100, 100], dropout=0, use_batchnorm=False,
                                  reg=reg, weight_scale=5e-2)
        solver = Solver(model, data,
                    num_epochs=30,
                    batch_size=100,
                    update_rule='adam',
                    optim_config={'learning_rate': lr},
                    lr_decay=0.95,
                    verbose=False)
        solver.train()
        
        val_accuracy = np.mean(solver.val_acc_history[-100:-1])
        if(val_accuracy>best_val):
            best_val = val_accuracy
            best_model = solver.model
            best_solver = solver

        plt.subplot(3, 1, 1)
        plt.title('Training loss')
        plt.xlabel('Iteration')
        plt.plot(solver.loss_history, 'o')

        plt.subplot(3, 1, 2)
        plt.title('Training accuracy')
        plt.xlabel('Epoch')
        plt.plot(solver.train_acc_history, '-o')

        plt.subplot(3, 1, 3)
        plt.title('Validation accuracy')
        plt.xlabel('Epoch')
        plt.plot(solver.val_acc_history, '-o')

        plt.gcf().set_size_inches(15, 15)
        print('counter: %d' %counter)
        fig = './graphs/history_%d'%counter
        plt.savefig(fig)
        plt.close()
        counter += 1



plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
plt.plot(best_solver.loss_history, 'o')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.plot(best_solver.train_acc_history, '-o')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
plt.plot(best_solver.val_acc_history, '-o')

plt.gcf().set_size_inches(15, 15)
plt.show()

""" Test """
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())






