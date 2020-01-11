
__precompile__()


using CSV
using DataFrames


function read_dataset(file_name="mnist_train")
   """
   Reads the training data from disk, converts the data from DataFrames to arrays
   Goes through the array, seperating the label and image data, which it returns as a list
   that contains tuples whose contents is the label and the image data.

   Attributes:

      * file_name (str) [default="mnist_train"]: File name of the CSV file (WITHOUT EXTENSION!)

   return: List of the data with tuples (label, image_array)
   """
    
   # Parse the file_path
   file_path = "data/$(lpad(file_name,2,"0")).csv"
   data = CSV.read(file_path)

   # Convert the dataframe to an array
   df1 = convert(Matrix, data)

   # Remove the first column
   image_array = df1[:, 1:size(df1,2) .!= 1] # Removes the first column
   answer_array = df1[:, 1] # Selects the first column

   # Inits empty image_list and sets the constraints of the image
   image_list = [];
   n_rows = 28
   n_cols = 28

   # Goes through the answer_array
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


function plot_picture(image)
   """
   Plots a picture given an array

   Attributes:

       * image (Vector): 1D input vector to reshape to a n*n matrix which will be plotted

   """

   # Takes the square of the matrix
   image_dim = sqrt(size(image)[1])

   # Checks if it is a perfect square or not
   if isinteger(image_dim) == true

       image_dim = Int(image_dim) # Convert Float to Int
       image = reshape(image, image_dim, image_dim) # Reshape the matrix to the squares dimensions
       heatmap(image) # Plot the image

   else

       throw("IMAGE_DIM_IS_FLOAT")

   end

end