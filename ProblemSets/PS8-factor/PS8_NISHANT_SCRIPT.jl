using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

include("lgwt.jl")
include("PS8_NISHANT_SOURCE.jl")

main()


# q2: are these correlations high? what might this imply?
# might imply that the ASVAB scores are highly correlated, which could suggest that they are measuring similar underlying abilities or skills. 
# This could have implications for factor analysis, as it may indicate that a single factor could explain a significant portion of the variance in these scores.