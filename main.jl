using Revise
using LinearAlgebra
using Distributions
#using Plots

# Check if the module is defined; if not --> Load the module
# if isdefined(Main, :Data_Reading) == false

#     println("Loading Data_Reading")
#     include("src/io_functions.jl")
#     using .Data_Reading

# end

# Plot the image
#heatmap(image)

# https://adventuresinmachinelearning.com/neural-networks-tutorial/


include("src/io_functions.jl")
include("src/activation_functions.jl")
include("src/neural_network.jl")

# Get the data
data = read_dataset(2)
label = data[1]
image = data[2]

# Normalisze the data
image = image / 255

const n_x = size(image)[1]
const n_h = 64
const output_size = 10
const learning_rate = 1

x = image

# Init the weights
w1 = rand(Truncated(Normal(0, 1), 0, 1), n_h, n_x)
w2 = rand(Truncated(Normal(0, 1), 0, 1), n_h, n_h)
w3 = rand(Truncated(Normal(0, 1), 0, 1), output_size, n_h)

# Init the biases
b1 = rand(Truncated(Normal(0, 1), 0, 1), n_h, 1)
b2 = rand(Truncated(Normal(0, 1), 0, 1), n_h, 1)

# Feedforward
z1 = (w1 * x) + b1
a1 = σ.(z1)

z2 = (w2 * a1) + b2
a2 = σ.(z2)

z3 = (w3 * a2)
a3 = σ.(z3)

desired_output = one_hot(label)
cost = MSE(a3, desired_output)

∂c_a = cost_derivative(a3, desired_output)
∂a_z = σ′.(z3)
∂z_w = z2

∂c_w = hadmard(∂c_a, ∂a_z)

#∂z_w * ∂a_z * 

# ∇a_C = cost_derivative(a3, desired_output)

# δ = hadmard(∇a_C, σ′.(z3))
# δ_1 = hadmard(transpose(w3) * δ, σ′.(z2))
# δ_2 = hadmard(transpose(w2) * δ_1, σ′.(z1)) 
\


println("---")