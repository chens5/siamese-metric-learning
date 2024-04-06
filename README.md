# Code for "Neural approximation of Wasserstein distance via a universal architecture for symmetric and factorwise group invariant functions"
## Requirements: 
PyTorch, Python OT, GeomLoss

## How to run
The autoencoder_model.py file contains all models: WPCE, Siamese DeepSets (under PointEncoder), and ProductNet.
We include the hyperparameter configurations we used in the `misc` folder as well as a model checkpoint for the ModelNet dataset with 1-Wasserstein distance (for point sets of size between 20 and 200. To generate your own hyperparameters, use the parameters.py script.
To train WPCE or Siamese DeepSets, use the train_autoencoder.py script.
To train ProductNet, use the train_productnet.py script


### Contact
Contact Samantha Chen (sac003@ucsd.edu) with questions
