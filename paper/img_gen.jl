using Revise
using LinearAlgebra
using Distributions
using Plots

# Change dir if not in it
if isdir("paper") == true
    cd("paper")
end

include("../src/io_functions.jl")
include("../src/activation_functions.jl")
include("../src/neural_network.jl")


# Plot sigmoid activation_function from -10:10 with a step size of 0.1
lower_bound = -10
upper_bound = 10
step_size = 0.1
file_name = "sigmoid_aktiveringfunktion"
file_path = "pictures/$(lpad(file_name,2,"0"))"

plot(σ, lower_bound:step_size:upper_bound, label = "\\sigma (x)" )
png(file_path)

# CNN gissningar array
#plot(1:100, prediction_arr, label="CNN", legend=:bottomright, yaxis="Antal korrekta gissningar / 10000", xaxis="Antal iterationer")

# Simple neural_network
#plot(1:100, prediction_arr, label="Simpelt neuronätverk", legend=:bottomright, yaxis="Antal korrekta gissningar", xaxis="Antal iterationer")


# p1= heatmap(reshape(train_data[1][2], 28, 28), xaxis=false, yaxis=false, titlefontsize=20, fontfamily=font(20, "MingLiU"), title=string(train_data[2][1]))
# p2 = heatmap(reshape(train_data[2][2], 28, 28), xaxis=false, yaxis=false, titlefontsize=20, fontfamily=font(20, "MingLiU"), title=string(train_data[2][1]))
# p3 = heatmap(reshape(train_data[3][2], 28, 28), xaxis=false, yaxis=false, titlefontsize=20, fontfamily=font(20, "MingLiU"), title=string(train_data[3][1]))
# p4 = heatmap(reshape(train_data[4][2], 28, 28), xaxis=false, yaxis=false, titlefontsize=20, fontfamily=font(20, "MingLiU"), title=string(train_data[4][1]))
# plot(p1,p2,p3,p4,layout=(2,2),legend=false)