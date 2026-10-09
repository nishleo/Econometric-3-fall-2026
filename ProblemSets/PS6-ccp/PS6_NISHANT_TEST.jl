
using Test, HTTP, GLM, LinearAlgebra, Random, Statistics
using DataFrames, DataFramesMeta, CSV

include(joinpath(@__DIR__, "create_grids.jl"))
include(joinpath(@__DIR__, "PS6_NISHANT_SOURCE.jl"))

@testset "Data Loading and Reshaping" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"

    df = load_and_reshape_data(url)

    # Check required columns
    @test all(x -> x in names(df),
              ["bus_id", "time", "Y", "Odometer", "RouteUsage", "Branded"])

    # Each bus should have 20 observations
    N = length(unique(df.bus_id))
    @test nrow(df) == N * 20

    # Time periods should range from 1 to 20
    @test minimum(df.time) == 1
    @test maximum(df.time) == 20

    # Every bus-time combination must be unique
    @test nrow(unique(df, [:bus_id, :time])) == nrow(df)

    # Data must be sorted by bus and time
    @test issorted(df.bus_id)

    # Check that decisions are binary
    @test all(y -> y in (0, 1), df.Y)

end


@testset "Flexible Logit" begin

    # Create balanced data to avoid perfect separation
    df = DataFrame(
        Odometer = Float64[],
        RouteUsage = Float64[],
        Branded = Int[],
        time = Float64[],
        Y = Int[]
    )

    for x in (0.0, 1.0, 2.0, 3.0),
        z in (0.25, 0.5, 0.75, 1.0),
        t in (1.0, 2.0, 3.0, 4.0),
        b in 0:1,
        y in 0:1

        push!(df, (x, z, b, t, y))
    end

    model = estimate_flexible_logit(df)

    # Check number of observations
    @test nobs(model) == nrow(df)

    # Seven interacted variables imply 2^7 coefficients
    @test length(coef(model)) == 128

    # Estimated coefficients must be finite
    @test all(isfinite, coef(model))

    # Probabilities must be between zero and one
    p = predict(model)
    @test all(x -> 0 <= x <= 1, p)

    # Balanced data should produce probabilities near 0.5
    @test all(isapprox.(p, 0.5, atol=1e-6))

end


@testset "State Space" begin

    xval = [0.0, 1.0, 2.0]
    zval = [0.25, 0.75]

    xbin = length(xval)
    zbin = length(zval)

    states = construct_state_space(xbin, zbin, xval, zval)

    # Number of states
    @test nrow(states) == xbin * zbin

    # Required columns
    @test names(states) ==
          ["Odometer", "RouteUsage", "Branded", "time"]

    # Mileage should repeat within each route block
    @test states.Odometer ==
          [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]

    # Route usage should remain constant within blocks
    @test states.RouteUsage ==
          [0.25, 0.25, 0.25, 0.75, 0.75, 0.75]

    # All state combinations should be unique
    @test nrow(unique(states, [:Odometer, :RouteUsage])) ==
          xbin * zbin

    # Initial brand and time values should be zero
    @test all(states.Branded .== 0)
    @test all(states.time .== 0)

end




@testset "Future Values" begin

    # Balanced data with known replacement probability of 0.5
    df = DataFrame(
        Odometer = Float64[],
        RouteUsage = Float64[],
        Branded = Int[],
        time = Float64[],
        Y = Int[]
    )

    for x in (0.0, 1.0, 2.0, 3.0),
        z in (0.25, 0.5, 0.75, 1.0),
        t in (1.0, 2.0, 3.0, 4.0),
        b in 0:1,
        y in 0:1

        push!(df, (x, z, b, t, y))
    end

    model = estimate_flexible_logit(df)

    xval = [0.0, 1.0]
    zval = [0.25, 0.75]

    states = construct_state_space(2, 2, xval, zval)

    transition = [1.0 0.0;
                  0.5 0.5;
                  1.0 0.0;
                  0.5 0.5]

    β = 0.9
    T = 3

    FV = compute_future_values(states, model,
                               transition, 2, T, β)

    # Check dimensions
    @test size(FV) == (4, 2, T + 1)

    # All future values must be finite
    @test all(isfinite, FV)

    # FV at time 1 is initialized to zero
    @test all(FV[:, :, 1] .== 0)

    # Known analytical value: -beta * log(0.5)
    expected = β * log(2)

    # With the t = 1:T-1 loop, slices 2:T are populated
    @test all(isapprox.(FV[:, :, 2:T], expected, atol=1e-6))

    # Last slice remains zero under this loop convention
    @test all(FV[:, :, T+1] .== 0)

    # Zero discount factor implies zero future value
    FV_zero = compute_future_values(states, model,
                                    transition, 2, T, 0.0)

    @test all(FV_zero .== 0)

end


@testset "Future Value Mapping" begin

    # Two buses observed for two periods
    df = DataFrame(
        bus_id = [1, 1, 2, 2],
        time = [1, 2, 1, 2],
        Branded = [0, 0, 1, 1]
    )

    Xstate = [2, 1, 2, 2]
    Zstate = [1, 1, 2, 2]

    xbin = 2

    # Each row sums to one: Markov transition matrix
    transition = [1.0 0.0;
                  0.25 0.75;
                  0.8 0.2;
                  0.1 0.9]

    # Check Markov property
    @test all(transition .>= 0)
    @test all(isapprox.(vec(sum(transition, dims=2)),
                        1.0, atol=1e-12))

    FV = zeros(4, 2, 3)

    FV[:, 1, 2] = [1.0, 5.0, 2.0, 12.0]
    FV[:, 1, 3] = [3.0, 11.0, 7.0, 27.0]
    FV[:, 2, 2] = [2.0, 8.0, 4.0, 34.0]
    FV[:, 2, 3] = [5.0, 15.0, 6.0, 46.0]

    fvt1 = compute_fvt1(df, FV, transition,
                         Xstate, Zstate, xbin)

    # Manually calculated expected values
    expected = [3.0, 0.0, 21.0, 28.0]

    @test length(fvt1) == nrow(df)
    @test fvt1 ≈ expected

    # Zero future values imply zero mapped values
    @test compute_fvt1(df, zeros(4, 2, 3),
                       transition, Xstate, Zstate, 2) ==
          zeros(4)

    # Doubling future values doubles mapped values
    @test compute_fvt1(df, 2 .* FV, transition,
                       Xstate, Zstate, 2) ≈ 2 .* expected

end


@testset "Structural Logit" begin

    # Balanced binary outcomes
    df = DataFrame(
        Odometer = Float64[],
        Branded = Int[],
        Y = Int[]
    )

    for x in (0.0, 1.0, 2.0, 3.0),
        b in 0:1,
        y in 0:1

        push!(df, (x, b, y))
    end

    # Zero offset
    fv = zeros(nrow(df))

    model = estimate_structural_params(copy(df), fv)

    # Intercept, mileage and brand coefficients
    @test length(coef(model)) == 3

    # Coefficients should be approximately zero
    @test coef(model) ≈ zeros(3) atol=1e-6

    # Fitted probabilities should be 0.5
    @test predict(model) ≈ fill(0.5, nrow(df)) atol=1e-6

    # Check that offset is correctly included
    fv_shifted = fill(0.7, nrow(df))

    model_shifted = estimate_structural_params(
        copy(df), fv_shifted
    )

    # Constant offset should shift intercept by -0.7
    @test coef(model_shifted) ≈ [-0.7, 0.0, 0.0] atol=1e-6

    # Probabilities should remain unchanged
    @test predict(model_shifted) ≈ predict(model) atol=1e-6

end




@testset "Grid Transition Matrix" begin

    zval, zbin, xval, xbin, xtran = create_grids()

    # Expected dimensions
    @test size(xtran, 1) == xbin * zbin
    @test size(xtran, 2) == xbin

    # Transition probabilities cannot be negative
    @test all(xtran .>= 0)

    # Each row must sum to one
    @test all(isapprox.(vec(sum(xtran, dims=2)),
                        1.0, atol=1e-8))

    # Check state-space dimensions
    states = construct_state_space(xbin, zbin, xval, zval)

    @test nrow(states) == size(xtran, 1)

end


println("\nAll PS6 unit tests completed.")
