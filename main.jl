using Revise
using LinearAlgebra
using Distributions


include("src/io_functions.jl")
include("src/activation_functions.jl")
include("src/neural_network.jl")


const n_x = 784 # Image as 1d vector
const n_h = 64 # Hidden layer size
const output_size = 10 # Number of output nodes
const η = 1 # Learning rate
const epoches = 2 # Number of training iteration


# Init the weights
let 

    # Generate the weights and biases with a gausian distribution
    μ = 0 # The mean of the truncated Normal
    _σ = 1  # The standard deviation of the truncated Normal

    # Init the weights
    w1 = rand(Truncated(Normal(μ, _σ), -1, 1), n_h, n_x)
    w2 = rand(Truncated(Normal(μ, _σ), -1, 1), n_h, n_h)
    w3 = rand(Truncated(Normal(μ, _σ), -1, 1), output_size, n_h)

    # Init the biases
    b1 = rand(Truncated(Normal(μ, _σ), -1, 1), n_h, 1)
    b2 = rand(Truncated(Normal(μ, _σ), -1, 1), n_h, 1)

    # Load the test and training data
    global train_data = read_dataset("mnist_train")
    global test_data = read_dataset("mnist_test")

    # Training variables
    old_cost = 0
    global correct_predictions = 0
    global cost_arr = []
    global prediction_arr = []

    activation_func="sigmoid"
    derative=true

    println("-- Starting --")

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
            a1 = activation_function(z1, activation_func)

            z2 = (w2 * a1) .+ b2
            a2 = activation_function(z2, activation_func)

            z3 = (w3 * a2)
            a3 = activation_function(z3, activation_func)

            # Make the label one hot encoded
            desired_output = one_hot(label)

            # Calculate the cost
            cost = MSE(a3, desired_output)

            # Back propigation
            # Error in output layer
            ∇a_C_3 = cost_derivative(a3, desired_output)
            δ_3 = hadmard(∇a_C_3, activation_function(z3, activation_func, derative))

            # Error in second layer
            ∇a_C_2 = (transpose(w3) * δ_3)  
            δ_2 = hadmard(∇a_C_2, activation_function(z2, activation_func, derative))

            # Error in first layer
            ∇a_C_1 = (transpose(w2) * δ_2)  
            δ_1 = hadmard(∇a_C_1, activation_function(z1, activation_func, derative))


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
            z1 = (w1 * x) .+ b1
            a1 = activation_function(z1, activation_func)

            z2 = (w2 * a1) .+ b2
            a2 = activation_function(z2, activation_func)

            z3 = (w3 * a2)
            a3 = activation_function(z3, activation_func)


            # Checks the prediction
            correct_predictions += prediction(a3, label)
            
        end

        println("Epoch: ", epoch, " | Number of correct_predictions: ", correct_predictions)
        append!(prediction_arr, correct_predictions)
        correct_predictions = 0

    end

end
