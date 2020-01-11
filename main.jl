using Revise
using LinearAlgebra
using Distributions
#using Plots

# https://adventuresinmachinelearning.com/neural-networks-tutorial/


include("src/io_functions.jl")
include("src/activation_functions.jl")
include("src/neural_network.jl")


const n_x = 784 # Image as 1d vector
const n_h = 64 # Hidden layer size
const output_size = 10 # Number of output nodes
const η = 0.1 # Learning rate
const epoches = 4


# Init the weights
let 

    w1 = rand(Truncated(Normal(0, 1), -1, 1), n_h, n_x)
    w2 = rand(Truncated(Normal(0, 1), -1, 1), n_h, n_h)
    w3 = rand(Truncated(Normal(0, 1), -1, 1), output_size, n_h)

    # Init the biases
    b1 = rand(Truncated(Normal(0, 1), -1, 1), n_h, 1)
    b2 = rand(Truncated(Normal(0, 1), -1, 1), n_h, 1)

    # Training variables
    old_cost = 0
    global correct_predictions = 0
    global cost_arr = []
    global train_data = read_dataset_2("mnist_train")
    global test_data = read_dataset_2("mnist_test")

    for epoch=1:epoches

        # Training
        for i=1:size(train_data)[1]

            # Get the data
            data = train_data[i]
            label = data[1]
            image = data[2]

            # Normalisze the data
            image = image / 255
            x = image

            # Feedforward
            z1 = (w1 * x) .+ b1
            a1 = σ.(z1)

            z2 = (w2 * a1) .+ b2
            a2 = σ.(z2)

            z3 = (w3 * a2)
            a3 = σ.(z3)

            # Make the label one hot encoded
            desired_output = one_hot(label)

            # Calculate the cost
            cost = MSE(a3, desired_output)
            #println(cost)
            #old_cost = cost
            #append!(cost_arr, cost)

            # Back propigation
            # Error in output layer
            ∇a_C_3 = cost_derivative(a3, desired_output)
            δ_3 = hadmard(∇a_C_3, σ′.(z3))

            # Error in second layer
            ∇a_C_2 = (transpose(w3) * δ_3)  
            δ_2 = hadmard(∇a_C_2, σ′.(z2))

            # Error in first layer
            ∇a_C_1 = (transpose(w2) * δ_2)  
            δ_1 = hadmard(∇a_C_1, σ′.(z1))


            # Gradient descent
            # Third layer
            w3 = w3 .- η * δ_3 * transpose(a2)

            # Second layer
            w2 = w2 .- η * δ_2 * transpose(a1)
            b2 = b2 .- η * δ_2

            # First layer
            w1 = w1 .- η * δ_1 * transpose(x)
            b1 = b1 .- η * δ_1

            if i % 10000 == 0
                println("Epoch : ", epoch, " | I: ", i)
            end

        end

        # Test
        for i=1:size(test_data)[1]

            # Get the data
            data = test_data[i]
            label = data[1]
            image = data[2]

            # Normalisze the data
            image = image / 255
            x = image

            # Feedforward
            z1 = (w1 * x) + b1
            a1 = σ.(z1)

            z2 = (w2 * a1) + b2
            a2 = σ.(z2)

            z3 = (w3 * a2)
            a3 = σ.(z3)

            # Checks the prediction
            correct_predictions += prediction(a3, label)

        end

        println("Epoch: ", epoch, " | Number of correct_predictions: ", correct_predictions)
        correct_predictions = 0

    end

end

# println("---")