
# Check if the module is defined; if not --> Load the module
if isdefined(Main, :Data_Reading) == false

    println("Loading Data_Reading")
    include("src/io_functions.jl")
    using .Data_Reading

end


image = read_dataset()
