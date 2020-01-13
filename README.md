
# Neural_network - Two neural networks written from scrach in Julia

This is a repository that contains two programs:

```feed_forward.jl``` - A simple backpropigating network  

```conv_network.jl``` -The simple backpropigating network but with a CNN 
layer in the beginning.

The network is trained on the MNIST dataset (60 000 training images) and 
validates on the test labels (10 000 images). The images are 28x28 
pixels.

# How to

1. Make sure you have Julia installed (> 1.0)

2. Make sure you have installed all required packages by pasting this 
into a Julia session:
```
using Pkg

Pkg.add("Distributions");Pkg.add("Random");Pkg.add("DataFrames");Pkg.add("CSV");Pkg.add("Revise");Pkg.add("Plots");

```

3. Make sure that the training and test data is located in the 
```data/``` folder.

4. Now you can run the program's and it will print out the number of 
correct predictions for each epoch.


