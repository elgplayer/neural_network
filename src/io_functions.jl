
using CSV
using PyPlot

file_path = "data/mnist_test_small.csv"
data = CSV.read(file_path)

# # Iterate over the rows
# for i = 1:size(data)[1]
#     #println(i)
#     row = data[i, :]
# end

i = 1
row = data[i, :]
label = row.label
picture_data = row[2:size(row)[1]]
