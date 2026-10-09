# Load required packages
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

include("create_grids.jl")
include("PS6_NISHANT_SOURCE.jl")

#run the main function from the source file
@time main()