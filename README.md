#  Mixture Density Network for predicting sequences

Given a sequence of data. The model creates a statistic over the next propable timestep depending on a given horizont of the past. This was the model tries to predict the next propable datapoint. This was enabled by a time embedding.

We are using a mixture density network (MDN, Bishop 1994) combined with a simple hierarchical structure analysis inside particular hidden layers.

## Getting Started

Decide for a sequence you want to predict. Create a textfile representing a numpy array of shape (N,1). 
Figure out the following parameters
* An appropriate lenght of the horizont for the model to observe in every state
* the amount of mixtures needed
* the amount of the hidden layer and layer of the closing MLP

You can also manipulate the length of the future horizont generated by the model (Attention: a future horizont of N results in training effort by a factor N), skipped data by the observation plus the training lenght and the number of timesteps that should be generated.

### Prerequisites

* tensorflow
* numpy
* matplotlib (optional if you don't want to see plots)

## Architecture
The computation graph consists of an Input, Filter, Hidden Layer, MLP, Lossfunction and Inference described below.
![graph](https://github.com/f37/MDN_music_MDP/blob/master/Architecture/graph.png)

### Filter
The filter consists of a fully connected layer which represents a learnable salientmap that results weight the input. This is a trivial approach of creating hierarchie structures.

### Input Layer
Classic. But filtered.

### Hidden Layer
Similar to the input particular hidden layers also contain a filter of the output of the previous layer. These should give the network the focus and weights over the datastructure of the input to that layer.
![graph](https://github.com/f37/MDN_music_MDP/blob/master/Architecture/graph.png)

### MLP and Output layer
Classic.

### Loss
Calculates the loss due to Bishop 1994.

### Inference
Creates a random variable for a given input from the resulting mixture parameter.

## Running

### Example usage

