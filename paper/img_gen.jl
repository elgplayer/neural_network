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

plot(σ, lower_bound:step_size:upper_bound, title="Sigmoid aktiveringsfunktion", label = "\\sigma (x)" )
png(file_path)

# CNN gissningar array
#plot(1:100, prediction_arr, label="CNN", legend=:bottomright, yaxis="Antal korrekta gissningar / 10000", xaxis="Antal iterationer")