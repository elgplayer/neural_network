
__precompile__()


#module Data_Reading

using CSV
using DataFrames

# # Iterate over the rows
# for i = 1:size(data)[1]
#     #println(i)
#     row = data[i, :]
# end
# https://www.juliabloggers.com/classifying-handwritten-digits-with-neural-networks/

#export read_dataset

function read_dataset(row=1, data_reshape=false)

   # Read the data
   #file_path = "data/mnist_test_small.csv"
   file_path = "data/mnist_train.csv"
   data = CSV.read(file_path)

   i = row
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


function read_dataset_2()
    
   file_path = "data/mnist_train.csv"
   data = CSV.read(file_path)

   # Convert the dataframe to an array
   df1 = convert(Matrix,data)

   # Remove the first column
   image_array = df1[:, 1:size(df1,2) .!= 1] # Removes the first column
   answer_array = df1[:, 1] # Selects the first column

   image_list = [];
   n_rows = 28
   n_cols = 28

   for i=1:size(answer_array)[1]

       image = image_array[i, :] # Select the entire row at index "i"
       image = reshape(image, n_rows, n_cols) # Reshape to a 28x28 array
       image = rotl90(image) # Rotate the image left 90 degress
       image = reshape(image, :, 1) # Convert back to a vector
       answer = answer_array[i] # Select answer

       # Create tuple
       image_item = (answer, image)

       # Push the item to the list
       push!(image_list, image_item)

   end

   return image_list

end



#end