using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

include("PS3_NISHANT_SOURCE.jl")

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
df, X, Z, y = load_data(url)

@testset "PS3 Tests" begin

    # Test data loading
    @test size(X, 2) == 3
    @test size(Z, 2) == 8
    @test length(y) == size(X, 1)

    # Test multinomial-logit likelihood
    theta_m = [zeros(21); 0.1]
    @test isfinite(mlogit_with_Z(theta_m, X, Z, y))

    # Test nested-logit likelihood
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
    theta_n = [zeros(6); 1.0; 1.0; 0.1]
    @test isfinite(nested_logit_with_Z(theta_n, X, Z, y, nesting_structure))

end

println("All tests passed.")