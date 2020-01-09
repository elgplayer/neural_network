
using CSV
#using PyPlot


file_path = "data/mnist_test_small.csv"
data = CSV.read(file_path)

# # Iterate over the rows
# for i = 1:size(data)[1]
#     #println(i)
#     row = data[i, :]
# end

# https://www.juliabloggers.com/classifying-handwritten-digits-with-neural-networks/

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
picture_data_matrix = reshape(picture_data_matrix, n_rows, n_cols)
   

