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

