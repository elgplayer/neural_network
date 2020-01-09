
__precompile__()


#module Data_Reading

using CSV


# # Iterate over the rows
# for i = 1:size(data)[1]
#     #println(i)
#     row = data[i, :]
# end
# https://www.juliabloggers.com/classifying-handwritten-digits-with-neural-networks/

#export read_dataset

function read_dataset(data_reshape=false)

   # Read the data
   file_path = "data/mnist_test_small.csv"
   data = CSV.read(file_path)

   i = 1
   row = data[i, :]
   label = row.label
   picture_data = row[2:size(row)[1]]

   picture_data_matrix = Vector{Float64}()
   for i=1:size(picture_data)[1]

      append!(picture_data_matrix, picture_data[i])

   end

   n_rows = 28
   n_cols = 28

   # Reshape the data
   picture_data_matrix = reshape(picture_data_matrix, n_rows, n_cols)

   # TODO: Unclear why
   # The data needs to be rotated 90 degress to the left
   picture_data_matrix = rotl90(picture_data_matrix)

   # TODO: This is retarded
   if data_reshape == false
      # Reverert back to vector format
      picture_data_matrix = reshape(picture_data_matrix, :, 1)
   end

   return (label, picture_data_matrix)

end


#end