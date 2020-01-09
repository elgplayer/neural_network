using Revise
using Plots

# Check if the module is defined; if not --> Load the module
# if isdefined(Main, :Data_Reading) == false

#     println("Loading Data_Reading")
#     include("src/io_functions.jl")
#     using .Data_Reading

# end

# Plot the image
#heatmap(image)



include("src/io_functions.jl")
include("src/activation_functions.jl")
include("src/neural_network.jl")

# Get the data
data = read_dataset()
label = data[1]
image = data[2]

# Normalisze the data
image = image / 255


const n_x = size(image)[1]
const n_h = 64
const output_size = 10
const learning_rate = 1

x = image

w1 = rand(n_h, n_x)
b1 = zeros(n_h, 1)

w2 = rand(1, n_h)
b2 = zeros(n_h, n_h)

w3 = rand(n_h, output_size) # No bias

z1 = (w1 * x) + b1
a1 = σ.(z1)
a1 = leaky_ReLU.(z1)

z2 = (a1 * w2) + b2
a2 = σ.(z2)
#a2 = leaky_ReLU.(z2)

z3 = a2 * w3
a3 = σ.(z3)
#a3 = leaky_ReLU.(z3)

oj = sum(a3, dims=1)
#println("oj")

#include("src/neural_network.jl")


println("---")