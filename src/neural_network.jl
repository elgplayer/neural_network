
# Y_𝒊 = [0, 0.2, 0.5, 0.9, 0, 0, 0, 0, 0.2, 0.53]
# Ŷ__𝒊 = [0, 0.0, 0.0, 1, 0, 0, 0, 0, 0.0, 0.0]

function MSE(Y_𝒊, Ŷ__𝒊)
    """
    Computes the MSE (Mean Squared Error)

    Attributes:

        * Y_𝒊 (Vector): Input Vector
        * Ŷ__𝒊 (Vector): Vector of the correct output (hot encoding)

    return: MSE sum
    """

    if length(Y_𝒊) == length(Ŷ__𝒊)

        return sum((Y_𝒊 - Ŷ__𝒊) .^ 2)
        
    else

        throw("DIMENSION_ERROR")

    end

end


function one_hot(label)
    """
    Takes a label and returns an array that is one hot encoded

    Attributes:
        
        * label (int): Label of the image that is the correct answer

    return: One hot encoded array
    """

    return_arr = zeros(10)
    return_arr[label + 1] = 1

    return return_arr

end