# Sigmoid activation function
σ(x::Real) = one(x) / (one(x) + exp(-x))
    
# Sigmoid derative
σ′(x) = σ(x) * (1 - σ(x))

# ReLU
function ReLU(x::Real)
    
    if x <= 0
    
        return 0
    
    else

        return x

    end

end

# Leaky ReLU
function leaky_ReLU(x::Real)
    
    if x < 0
    
        return 0.01 * x
    
    else

        return x
    
    end

end

# Relu derative
function ReLU_der(x::Real)

    if x < 0

        return 0

    else

        return 1

    end

end


function activation_function(input, derative=false, activation_function="sigmoid")

    if activation_function == "sigmoid"

        if derative == true

            return σ′.(input)

        else

            return σ.(input)

        end


    elseif activation_function == "ReLU"

        if derative == true

            return ReLU.(input)

        else

            return ReLU_der.(input)

        end

    
    elseif  activation_function == "leaky_ReLU"

        if derative == true

            return leaky_ReLU.(input)

        else

            return ReLU_der.(input)

        end

    else

        throw("INVALID_ACTIVATION_FUNCTION")

    end

end