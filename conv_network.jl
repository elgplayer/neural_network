using CSV
using DataFrames

using Revise
using LinearAlgebra
using Distributions
using Random

include("src/io_functions.jl")
include("src/activation_functions.jl")
include("src/neural_network.jl")
include("src/conv_operations.jl")




const n_x = 1352 # Image as 1d vector
const n_h = 64 # Hidden layer size
const output_size = 10 # Number of output nodes

let

    Random.seed!(1234) # Random seeed
    η = 0.01 # Learning rate
    epoches = 2 # Number of training iteration

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


    mutable struct Weights
        w1; w2; w3;
    end

    mutable struct Bias
        b1; b2;
    end

    # Make the weights and biases global
    global weights = Weights(w1, w2, w3)
    global bias = Bias(b1, b2)


    # Load the test and training data
    global train_data = read_dataset("mnist_train")
    global test_data = read_dataset("mnist_test")

    # Training variables
    old_cost = 0
    global correct_predictions = 0
    global cost_arr = []
    global prediction_arr = []
    global wrong_predictions = []


    activation_func="sigmoid"
    derative=true

    println("-- Starting CNN --")

    @time begin
    for epoch=1:epoches

        if epoch == epoches
            save_fails = true
        else
            save_fails = false
        end

        # Test
        for i=1:size(test_data)[1]

            # Get the data
            data = train_data[i]
            label = data[1]
            image = data[2]

            # Normalisze the data
            image = image / 255
            x = image
            x_matrix = reshape(x, 28, 28)
            
            number_of_filters = 8
            filter_size = 3

            # Init the data with a gausian distribution
            conv_filters = rand(Truncated(Normal(μ, _σ), -1, 1), filter_size, filter_size, number_of_filters,) / 9
            conv_arr = []
            pooling_arr = []

            # Convelution
            for a=1:number_of_filters

                # Select the filter that was randomly generated
                filter = conv_filters[:, :, a]

                # Stride the filter over te image
                strided = striding(x_matrix, filter)

                # Append the strided array to an array
                append!(conv_arr, strided)

            end
            # Reshape the filter to a 26x26x8 matrix
            conv_arr = reshape(conv_arr, 26, 26, number_of_filters)

            # Pooling
            for a=1:number_of_filters

                append!(pooling_arr, pooling(conv_arr[:, :, a]))

            end
            # Reshape the filter to a 13x13x8 matrix
            pooling_arr = reshape(pooling_arr, 13, 13, 8)

            x = vcat(pooling_arr...)

            # Feedforward
            z1 = (weights.w1 * x) .+ bias.b1
            a1 = activation_function(z1, activation_func)

            z2 = (weights.w2 * a1) .+ bias.b2
            a2 = activation_function(z2, activation_func)

            z3 = (weights.w3 * a2)
            a3 = activation_function(z3, activation_func)

            # Checks the prediction
            correct_predictions += check_prediction(a3, label, save_fails, image)
            
        end
    
        for i=1:size(train_data)[1]

            # Get the data
            data = train_data[i]
            label = data[1]
            image = data[2]

            # Normalisze the data
            image = image / 255
            x = image
            x_matrix = reshape(x, 28, 28)

            number_of_filters = 8
            filter_size = 3

            # Init the data with a gausian distribution
            conv_filters = rand(Truncated(Normal(μ, _σ), -1, 1), filter_size, filter_size, number_of_filters,) / 9
            conv_arr = []
            pooling_arr = []

            # Convelution
            for a=1:number_of_filters

                # Select the filter that was randomly generated
                filter = conv_filters[:, :, a]

                # Stride the filter over te image
                strided = striding(x_matrix, filter)

                # Append the strided array to an array
                append!(conv_arr, strided)

            end
            # Reshape the filter to a 26x26x8 matrix
            conv_arr = reshape(conv_arr, 26, 26, number_of_filters)

            # Pooling
            for a=1:number_of_filters

                append!(pooling_arr, pooling(conv_arr[:, :, a]))

            end
            # Reshape the filter to a 13x13x8 matrix
            pooling_arr = reshape(pooling_arr, 13, 13, 8)


            x = vcat(pooling_arr...)

            # Feedforward
            z1 = (weights.w1 * x) .+ bias.b1
            a1 = activation_function(z1, activation_func)

            z2 = (weights.w2 * a1) .+ bias.b2
            a2 = activation_function(z2, activation_func)

            z3 = (weights.w3 * a2)
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
            ∇a_C_2 = (transpose(weights.w3) * δ_3)  
            δ_2 = hadmard(∇a_C_2, activation_function(z2, activation_func, derative))

            # Error in first layer
            ∇a_C_1 = (transpose(weights.w2) * δ_2)  
            δ_1 = hadmard(∇a_C_1, activation_function(z1, activation_func, derative))


            # Gradient descent
            # Third layer
            weights.w3 = weights.w3 .- η * δ_3 * transpose(a2)

            # Second layer
            weights.w2 = weights.w2 .- η * δ_2 * transpose(a1)
            bias.b2 = bias.b2 .- η * δ_2

            # First layer
            weights.w1 = weights.w1 .- η * δ_1 * transpose(x)
            bias.b1 = bias.b1 .- η * δ_1


            if i % 10000 == 0
                println("Epoch : ", epoch, " | I: ", i)
            end

        end
            
        println("Epoch: ", epoch, " | Number of correct_predictions: ", correct_predictions)
        append!(prediction_arr, correct_predictions)
        correct_predictions = 0

    end
    end

end