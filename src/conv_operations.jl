

#input_matrix = [2 0 1 1; 0 1 0 0; 0 0 1 0; 0 3 0 0]
#filter_matrix = [1 0 1; 0 0 0; 0 1 0]

# Striding: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
# Adding the result to a new matrix: https://stackoverflow.com/questions/29897498/how-to-construct-matrix-from-row-column-vectors-in-julia

#input_matrix = [0 0 0 0 0; 0 0 3 6 0; 0 1 4 7 0; 0 2 5 8 0; 0 0 0 0 0]
#filter_matrix = [0 2; 1 3]

input_matrix = [2 6 3 7 4 3  0; 3 6 4 8 2 2 1; 7 9 8 3 1 4 3; 4 8 3 6 8 1 9; 6 7 8 6 3 9 2; 2 4 9 3 4 8 1; 9 3 7 4 6 3 4]
filter_matrix = [3 1 -1; 4 0 0; 4 2 3]

# Todo: Maybe support diffrent stride for X and Y
function striding(input_arr, filter_arr, step_size=1)
    #=

    Stride over an array given an kernel

    input_arr: Input matrix
    filter_arr: Kernel
    step_size: How much should the filter move?

    return: New smaller array

    =#
    println("-------")

    # Gets the box size
    input_arr_size = size(input_arr, 2)
    filter_arr_size = size(filter_arr, 2)

    
    number_of_steps = (input_arr_size - filter_arr_size)/step_size + 1 # How many times should we stride in one direction
    index_offset = [0, 0] # Init the index off Set
    new_arr = Vector{Float64}() # Empty 1-d Vector

    # Gets the Temporary arrys first index
    first_tmp_index = 1 + index_offset[1] + index_offset[2] * input_arr_size

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

            #break

        end

    end


    # Reshapes the 1-d vector into a 2-d matrix the size of number_of_steps*number_of_steps
    new_arr = reshape(new_arr, Int(number_of_steps), :)

    return new_arr

end


function temp_index(input_arr, filter_arr, index_offset)
    #=

    Creates a temporary array that can be used to do the elementwise multiplication with the kernel

    input_arr: Input matrix
    filter_arr: Kernel matrix
    index_offset: How offset the index is

    =#

    # Gets the size of the arrays
    input_arr_size = size(input_arr, 2)
    filter_arr_size = size(filter_arr, 2)

    # Calculate the index of the upper right corner of the filter
    x = index_offset[1] * input_arr_size
    y = index_offset[2]
    matrix_index = x + y + 1

    # Init empty array at the size of the filter
    tmp_arr = Matrix{Union{Nothing, Int64}}(nothing, filter_arr_size, filter_arr_size)

    # Loop through the filter
    for i = 0:filter_arr_size-1

        # Calculates the position of the filter in the input_arr
        tmp_arr_i = i+1
        current_i = matrix_index + i

        tmp_arr[tmp_arr_i] = input_arr[current_i]

        # ???
        tmp_arr_i_2 = tmp_arr_i + 1 * filter_arr_size
        current_i_2 = tmp_arr_i_2 + 1 * input_arr_size

        # This calculate the filter to the right of the left column
        for α = 1:filter_arr_size-1

            tmp_arr_i_2 = tmp_arr_i + α * filter_arr_size
            current_i_2 = current_i + α * input_arr_size

            tmp_arr[tmp_arr_i_2] = input_arr[current_i_2]

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

oi = striding(input_matrix, filter_matrix, 2)
