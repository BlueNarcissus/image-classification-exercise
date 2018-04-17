Overall structure of fully-connceted neural network

1. 'FullyConnected.py': main script to train a model.

2. 'cs231n/classifiers/fc_net.py': training model for an L-layer neural network.
   'cs231n/layers.py': functions to compute loss and gradients. 

3. 'cs231n/optim.py': different gradient descent update rules, connected to 'FullyConnected.py' by 'Solver.py', 
    which is well-written in the course materials.
    
    Update rules include:
      - SGDStochastic Gradient Descent
      - SGD+Momentum
      - RMSProp
      - Adam

Results:
- Adam is the update rule that achieves the best training and validation accuracy.
- The best validation accuracy achieved so far is around 55%.
