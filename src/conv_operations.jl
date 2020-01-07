
import Statistics

input_matrix = [2 0 1 1; 0 1 0 0; 0 0 1 0; 0 3 0 0]
filter_matrix = [1 0 1; 0 0 0; 0 1 0]

# Striding: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
# Adding the result to a new matrix: https://stackoverflow.com/questions/29897498/how-to-construct-matrix-from-row-column-vectors-in-julia

# input_matrix = [0 0 0 0 0; 0 0 3 6 0; 0 1 4 7 0; 0 2 5 8 0; 0 0 0 0 0]
# filter_matrix = [0 2; 1 3]

#input_matrix = [2 6 3 7 4 3  0; 3 6 4 8 2 2 1; 7 9 8 3 1 4 3; 4 8 3 6 8 1 9; 6 7 8 6 3 9 2; 2 4 9 3 4 8 1; 9 3 7 4 6 3 4]
#filter_matrix = [3 1 -1; 4 0 0; 4 2 3]

input_matrix = [2 3 0 5 2.5 0; 2 1.5 0.5 0 7 0; 1.5 5 5 3 2 0; 3 5 7 1.5 0 0 ; 2 5 2 1.5 2 0; 0 0 0 0 0 0]

# Todo: Maybe support diffrent stride for X and Y
function striding(input_matrix, filter_arr, step_size=1)
    #=

    Stride over an array given an kernel

    input_matrix: Input matrix
    filter_arr: Kernel
    step_size: How much should the filter move?

    return: New smaller array

    =#
    println("-------")

    # Gets the box size
    input_matrix_size = size(input_matrix, 2)
    filter_arr_size = size(filter_arr, 2)
    
    number_of_steps = (input_matrix_size - filter_arr_size)/step_size + 1 # How many times should we stride in one direction
    index_offset = [0, 0] # Init the index off Set
    new_arr = Vector{Float64}() # Empty 1-d Vector

    # Gets the Temporary arrys first index
    first_tmp_index = 1 + index_offset[1] + index_offset[2] * input_matrix_size

    # Creates the index_offset_list
    for y=0:number_of_steps-1

        # Set the index offset
        x = 0
        index_offset[1] = x * step_size
        index_offset[2] = y * step_size

        # Calculate the temporary array
        tmp_arr = temp_index(input_matrix, filter_matrix, index_offset)
        
        # Do the element wise multiplication with the kernel
        Σ_filter = element_wise_mult(tmp_arr, filter_matrix)
        
        # Append the sum to a 1-d vector
        append!(new_arr, Σ_filter)

        # Iterates over the X-axis
        for x = 1:number_of_steps-1

            index_offset[1] = x * step_size
            index_offset[2] = y * step_size

            # Calculate the temporary array
            tmp_arr = temp_index(input_matrix, filter_matrix, index_offset)
            
            # Do the element wise multiplication with the kernel
            Σ_filter = element_wise_mult(tmp_arr, filter_matrix) 

            # Append the sum to a 1-d vector
            append!(new_arr, Σ_filter)

        end

    end


    # Reshapes the 1-d vector into a 2-d matrix the size of number_of_steps*number_of_steps
    new_arr = reshape(new_arr, Int(number_of_steps), :)

    return new_arr

end


function temp_index(input_matrix, filter_arr, index_offset)
    #=

    Creates a temporary array that can be used to do the elementwise multiplication with the kernel

    input_matrix: Input matrix
    filter_arr: Kernel matrix
    index_offset: How offset the index is

    =#

    # Gets the size of the arrays
    input_matrix_size = size(input_matrix, 2)
    filter_arr_size = size(filter_arr, 2)

    # Calculate the index of the upper right corner of the filter
    x = index_offset[1] * input_matrix_size
    y = index_offset[2]
    matrix_index = x + y + 1

    # Init empty array at the size of the filter
    tmp_arr = Matrix{Union{Nothing, Int64}}(nothing, filter_arr_size, filter_arr_size)

    # Loop through the filter
    for i = 0:filter_arr_size-1

        # Calculates the position of the filter in the input_matrix
        tmp_arr_i = i+1
        current_i = matrix_index + i

        tmp_arr[tmp_arr_i] = input_matrix[current_i]

        # ???
        tmp_arr_i_2 = tmp_arr_i + 1 * filter_arr_size
        current_i_2 = tmp_arr_i_2 + 1 * input_matrix_size

        # This calculate the filter to the right of the left column
        for α = 1:filter_arr_size-1

            tmp_arr_i_2 = tmp_arr_i + α * filter_arr_size
            current_i_2 = current_i + α * input_matrix_size

            tmp_arr[tmp_arr_i_2] = input_matrix[current_i_2]

        end

    end

    return tmp_arr

end


function element_wise_mult(tmp_arr, filter_matrix)
    #=

    Does the element wise multiplication

    tmp_arr: Input array
    filter_matrix: kernel

    return: Sum of the given element wise multiplication

    =#

    Σ_filter = 0

    for (i,v) in enumerate(tmp_arr)

        Σ_filter += tmp_arr[i] * filter_matrix[i]

    end

    return Σ_filter

end


function create_tmp_array(input_matrix, matrix_index, pool_size=(2,2))

    input_matrix_size = size(input_matrix, 2)

    new_arr = Vector{Float64}() # Empty 1-d Vector

    println("Matrix index: ", matrix_index)

    # Y-axis
    for y = 0:pool_size[2]-1

        input_matrix_i = y + matrix_index
        println("input i: ", input_matrix_i)

        # Get the value from the input_matrix
        arr_value = input_matrix[input_matrix_i]

        # Append the sum to a 1-d vector
        append!(new_arr, arr_value)

        for x=1:pool_size[1]-1

            input_matrix_i = x * input_matrix_size + matrix_index + y
            #println("input i: ", input_matrix_i)

            # Get the value from the input_matrix
            arr_value = input_matrix[input_matrix_i]

            # Append the sum to a 1-d vector
            append!(new_arr, arr_value)

        end

    end

    #show(stdout, "text/plain", new_arr)

    # Reshapes the 1-d vector into a 2-d matrix the size of number_of_steps*number_of_steps
   #new_arr = reshape(new_arr, Int(pool_size[1]), :)

    return new_arr


end


function calculate_pool_value(tmp_array, pooling_type)
    """
    Calculate the value of the tmp_array given a pooling type

    Attributes:
        
        * tmp_array (array): Temporary array
        * pooling_type (str): What type of pooling (Max or Mean)

    return: Pool value calculated from the tmp_array
    """
    pool_value = Nothing
    
    if pooling_type == "max"

        pool_value = maximum(tmp_array)
    
    elseif pooling_type == "mean"

        pool_value = Statistics.mean(tmp_array)

    else

        throw("Pooling function not supported")

    end

    return pool_value

end


function pooling(input_matrix, pool_size=(2,2), pooling_type="max", pad=false)
    """
    
    Non overlapping pooling

    Attributes:

        * input_matrix (2d Matrix): Input matrix to perform the pooling one
        * pool_size (tuple): Tuple which shows the size of the pooling box in the X and Y direction
        * pooling_type (str): What type of pooling? (Max or Mean)
        * pad (boolean): Should we add zeros on the edge?

    """


    debug = true
    new_array = Vector{Float64}()
    

    if debug == true
        
        println("----")
        println("Input matrix: ")
        show(stdout, "text/plain", input_matrix)    
        println("\n")
    
    end

    # Get the size of the input_matrix
    input_matrix_size = size(input_matrix, 2)

    # Create a tuple of how many steps the filter can go
    number_of_steps = (input_matrix_size / pool_size[1], input_matrix_size / pool_size[2])
    index_offset = [0, 0] # Init the index off Set

    # Check if the number of steps are divisible
    if isinteger(number_of_steps[1]) == false || isinteger(number_of_steps[2])  == false

        if pad == true

            println("Starting to pad!")

        else

            throw("FILTER_IS_FLOAT")

        end

    end

    # Creates the index_offset_list
    for x = 0:number_of_steps[1]-1

        # Set the index offset
        index_offset[1] = x * input_matrix_size * pool_size[1] 

        for y=0:number_of_steps[2]-1

            index_offset[2] = y * pool_size[2] + 1

            matrix_index = index_offset[1] + index_offset[2]
            
            # Append the new pool value to the temporary array
            tmp_arr = create_tmp_array(input_matrix, matrix_index, pool_size)
            pool_value = calculate_pool_value(tmp_arr, pooling_type)
            append!(new_array, pool_value)

        end

    end


    if debug == true

        println("Output matrix: ")
        show(stdout, "text/plain", new_array)
        println("\n")

    end

    # If it is a square, then we can resize it
    if pool_size[1] == pool_size[2]

        new_array = reshape(new_array, Int(sqrt(size(new_array)[1])), :)

    end


    return new_array
    
end



new_array = pooling(input_matrix, (2, 2), "max")



# Shows the convuluted array
# smaller_array = striding(input_matrix, filter_matrix, 2)
# show(stdout, "text/plain", smaller_array)ö