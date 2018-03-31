from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.data_utils import load_CIFAR10

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """ Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the two-layer neural net classifier. """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    # validation set
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # training set
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # test set
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

""" Train neural network: SGD with momentum """
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10

from cs231n.vis_utils import visualize_grid
# Visualize the weights of the network
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    fig = './note/neural/train/weights_%d%d%d.png'%(i,j,k)
    plt.savefig(fig)
    plt.close()
    #plt.show()


""" Tune hyperparameters to get the best model """
results = {}
best_val = -1
best_net = None # store the best model into this
best_model = {}

learning_rate_decay = 0.95
# toy
learning_rate=1e-4
reg=0.25
num_iters=1000

#set1: best(lr=1e-3, reg=0.3, 55), val_acc = 50.5%, test_acc = 50.2%
learning_rates = [6e-4, 8e-4, 9e-4, 1e-3, 1.1e-3]
regularization_strengths = [0.28, 0.3, 0.33, 0.35, 0.4, 0.45]
hidden_units = [53, 55, 58]
num_iterations = [1600]

# set2: best(lr=1.05e-3, reg=0.33, 56), val_acc = 51.6%, test_acc = 49.1%
learning_rates = [9e-4, 9.5e-4, 1e-3, 1.05e-3]
regularization_strengths = [0.3, 0.33, 0.35, 0.4]
hidden_units = [55, 56, 57]
num_iterations = [1600]

i=0
for lr in learning_rates:
    i=i+1
    j=0
    for reg in regularization_strengths:
        j=j+1
        k=0
        for hidden_size in hidden_units:
            k = k+1
            for  num_iters in num_iterations:
                net = TwoLayerNet(input_size, hidden_size, num_classes)
                # Train the network
                stats = net.train(X_train, y_train, X_val, y_val, lr, learning_rate_decay, reg, num_iters, batch_size=200, verbose=True)
                                  
                train_accuracy = np.mean(net.predict(X_train)==y_train)
                val_accuracy = np.mean(net.predict(X_val)==y_val)
                results[(lr, reg, hidden_size,  num_iters)] = (train_accuracy, val_accuracy)
        
                if val_accuracy>best_val:
                    best_val=val_accuracy
                    best_net=net
                    best_model['lr'], best_model['reg'] = lr, reg
                    best_model['hidden_size']=hidden_size
                    best_model['num_iterations']=num_iters
        
                print('lr: %e reg: %e hidden size: %d num_iterations: %d' %(lr, reg, hidden_size, num_iters))
                print('training accuracy: %f' %train_accuracy)
                print('validation accuracy: %f' %val_accuracy)
                print('best value accuracy: %f' %best_val)

                """ Debug the training """
                # Plot the loss function and train / validation accuracies
                plt.subplot(2, 1, 1)
                plt.plot(stats['loss_history'])
                plt.title('Loss history')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                axes = plt.gca()
                axes.set_ylim([1.2, 2.4])


                plt.subplot(2, 1, 2)
                plt.plot(stats['train_acc_history'], label='train')
                plt.plot(stats['val_acc_history'], label='val')
                plt.title('Classification accuracy history')
                plt.xlabel('Epoch')
                plt.ylabel('Clasification accuracy')
                axes = plt.gca()
                axes.set_ylim([0, 0.6])
                fig = './note/neural/train/track_%d%d%d.png'%(i,j,k)
                plt.savefig(fig)
                plt.close()
                #plt.show()

                # visualize the weights
                show_net_weights(net)


for lr, reg, hidden_size, num_iters in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, hidden_size, num_iters)]
    print('(lr %e reg %e hidden units %d), (train accuracy: %f val accuracy: %f)' % (lr, reg, hidden_size, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# visualize the weights of the best network
show_net_weights(best_net)

""" Run on the test set """
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)

