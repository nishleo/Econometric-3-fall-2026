using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables

Random.seed!(1234)
include("PS7_NISHANT_SOURCE.jl")
main()

# 2(c)
# Is the GMM objective function globally concave?
# No, it doesn't appear to be, because we got wildly different estimates
# from the same objective function with different starting values.
# BUT the MLE objective function is globally concave for this model!!!