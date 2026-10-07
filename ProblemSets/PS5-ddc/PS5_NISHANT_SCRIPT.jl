using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Read in function to create state transitions for dynamic model
include("create_grids.jl")

include("PS5_NISHANT_SOURCE.jl")

# run the main function
main()