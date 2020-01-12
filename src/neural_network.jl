using Plots


function MSE(Y_𝒊, Ŷ__𝒊)
    """
    Computes the MSE (Mean Squared Error)

    Attributes:

        * Y_𝒊 (Vector): Input Vector
        * Ŷ__𝒊 (Vector): Vector of the correct output (hot encoding)

    return: MSE sum
    """

    if length(Y_𝒊) == length(Ŷ__𝒊)

        return (sum((Y_𝒊 - Ŷ__𝒊) .^ 2) / 2)
        
    else

        throw("DIMENSION_ERROR")

    end

end


function cost_derivative(output, desired_output)
    """
    Derative of the cost function MSE

    Attributes:

        * output (Vector): What the network predicated
        * desired_output (Vector): What the output should have been

    return: Derative of the MSE (Vector)
    """

    return (output - desired_output)

end


function hadmard(A, B)
    """
    Creates a new array by taking the hadmard product of two arrays (element wise multiplication)

    Attributes:

        A (matrix): Matrix A
        B (matrix): Matrix B 

    return: New array as hadmard product of the array A and B
    """
    return broadcast!(*,A,A,B)

end


function one_hot(label)
    """
    Takes a label and returns an array that is one hot encoded

    Attributes:
        
        * label (int): Label of the image that is the correct answer

    return: One hot encoded array
    """

    return_arr = zeros(10) # Create zero array
    return_arr[label + 1] = 1 # Add a 1 at the index of the label

    return return_arr

end


function check_prediction(output, label, save_fails, image)
    """
    Checks if the prediction of the network (maxium value of the output)
    matches with the label

    Attributes:
        
        * output (Vector): Output of the network
        * label (int): The correct output

    return: 1 if the answer is correct; else return 0
    """
    
    # Checks if the index of the output corresponds to the correct label
    prediction = findmax(output)[2][1]-1

    if (prediction) == label

        return 1

    else

        if save_fails == true
            output_tuple = (prediction, label, image)
            push!(wrong_predictions, output_tuple)
        end
        
        return 0

    end

end




