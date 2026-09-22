using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

include("PS3_NISHANT_SOURCE.jl")

# call main function
allwrap()

# interpretation of gamma hat
# change in utility if i increase my wage by 1 unit, holding all else constant.
# gamma hat = -.09
# people dont like money
# model is misspecified, its bad.
# probably missing unobserved attributes like work environment, commute time, etc. that are correlated with wages.