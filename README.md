#  Sequence predicting Mixture Density Network (MDN)

Given a sequence of data. The model creates a statistic over the next 
probable time step depending on a given horizon of the past. The model 
tries to predict the next probable data point.

We are using a mixture density network (MDN, Bishop 1994) combined with a 
simple hierarchical structure analysis inside particular hidden layers. The
 trainingdata will be preprocessed and time embedded.

## Getting Started

Decide for a sequence you want to predict. Create a textfile representing a
 numpy array of shape (N,1). 
Figure out the following parameters
* An appropriate lenght of the horizont for the model to observe in every 
state
* the amount of mixtures needed
* the amount of the hidden layer and layer of the closing MLP

You can also manipulate the length of the future horizont generated by the 
model (Attention: a future horizont of N results in training effort by a 
factor N), skipped data by the observation plus the training lenght and the
 number of timesteps that should be generated.

### Prerequisites

* tensorflow
* numpy
* matplotlib (optional if you don't want to see plots)

## Architecture
The computation graph consists of an Input, Filter, Hidden Layer, MLP, 
Lossfunction and Inference described below.
![graph](https://github.com/f37/MDN_music_MDP/blob/master/Architecture/graph.png)

### Filter
The filter consists of a fully connected layer which represents a learnable
 salientmap that results weight the input. This is a trivial approach of 
 creating hierarchie structures.

### Input Layer
Classic. But filtered.

### Hidden Layer
Similar to the input particular hidden layers also contain a filter of the 
output of the previous layer. These should give the network the focus and 
weights over the datastructure of the input to that layer.
![hidden](https://github.com/f37/MDN_music_MDP/blob/master/Architecture/hidden.png)

### MLP and Output layer
Classic.

### Loss
Calculates the loss due to Bishop 1994.

### Inference
Creates a random variable for a given input from the resulting mixture 
parameter. In this project for convenience the inference chooses a mixture 
randomly by their occurance propability Pi and gives the corresponding Mu 
as output
![hidden](https://github.com/f37/MDN_music_MDP/blob/master/Architecture/inference.png)

## Running

Example files are provided, but feel free to use your own. The model 
creates a statistic from the data i.e. a distribution depending on the 
input data. From that it autonomously generates next propable outputs. This
 way we can create sort of music (see below)

### Exampledata folders
#### Example

Provides various signals. Depending on the length the training can last for
 ours. A Toyexample is "sinus.txt" that can show you the proof of work of 
 the algorithm.

#### MIDI

Contains midi files and a corresponding numpy file of the right format for 
the model. A converter (python2) is provided by "midi_conf.py". This will 
take as input the filepath (.mid or .np) and the resulting destination path.

The converter works for both directions. Be aware that midi often works 
with different tracks and this version only supports one track without 
prediction of the tone lenght.
