using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
using ForwardDiff
# Include quadrature function (make sure lgwt.jl is in your working directory)
include("lgwt.jl")

include("PS4_NISHANT_SOURCE.jl")

# call our function
allwrap()
