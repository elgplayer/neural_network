using Revise

# Check if the module is defined; if not --> Load the module
# if isdefined(Main, :Data_Reading) == false

#     println("Loading Data_Reading")
#     include("src/io_functions.jl")
#     using .Data_Reading

# end

include("src/io_functions.jl")


using Plots

# Get the data
data = read_dataset()
label = data[1]
image = data[2]

# Normalisze the data
image = image / 255

# Plot the image
heatmap(image)

println("oj")

#include("src/neural_network.jl")