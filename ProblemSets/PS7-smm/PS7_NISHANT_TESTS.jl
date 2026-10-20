using Test
using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables


include("PS7_NISHANT_SOURCE.jl")

# Independent reference probabilities. The final alternative is normalized to zero.
function reference_probabilities(α, X)
    K = size(X, 2)
    @assert length(α) % K == 0
    β = hcat(reshape(α, K, length(α) ÷ K), zeros(K))
    V = X * β
    V = V .- maximum(V; dims=2)
    P = exp.(V)
    return P ./ sum(P; dims=2)
end

function choice_indicators(y, J)
    return hcat([(y .== j) for j in 1:J]...)
end

function small_logit_fixture()
    X = hcat(ones(9),
             [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
             [0.4, -0.5, 1.0, -1.0, 0.7, -0.2, 1.2, -0.7, 0.3])
    y = [1, 2, 3, 2, 1, 3, 1, 2, 3]
    α = [0.3, -0.2, 0.15, -0.4, 0.25, -0.12]  # K=3, J=3
    return X, y, α
end

function simulation_coefficients()
    # Exactly the hcat example in the user's source for J=4.
    return hcat([1.0, -1.0, 0.5],
                [-2.0, 0.5, 0.3],
                [0.0, -0.5, 2.0],
                zeros(3))
end


@testset "Data loading: source method exists (online check is opt-in)" begin
    @test hasmethod(load_data, Tuple{String})
end

@testset "Data preparation: occupation recoding and X construction" begin
    occ = Union{Missing, Int}[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, missing]
    df0 = DataFrame(occupation=occ,
                    age=collect(21:34),
                    race=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                    collgrad=[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    df, X, y = prepare_occupation_data(df0)
    @test nrow(df) == 13
    @test size(X) == (13, 4)
    @test y == vcat(collect(1:7), fill(7, 6))
    @test all((y .>= 1) .& (y .<= 7))
    @test X[:, 1] == ones(13)
    @test X[:, 2] == df.age
    @test X[:, 3] == Float64.(df.race .== 1)
    @test X[:, 4] == df.collgrad
    @test df.white == (df.race .== 1)
    @test all(.!ismissing.(df.occupation))
    @test size(X, 1) == length(y)
end

@testset "Q1: OLS identity-weighted GMM objective" begin
    Random.seed!(101)
    N = 180
    X = hcat(ones(N), randn(N), randn(N), randn(N))
    βtrue = [1.2, -0.6, 0.8, -0.3]
    y = X * βtrue .+ 0.2 .* randn(N)
    βols = X \ y

    @test ols_gmm(βtrue, X, y) ≈ sum(abs2, y - X * βtrue)
    @test ols_gmm(βols, X, y) ≈ sum(abs2, y - X * βols)
    @test ols_gmm(βols, X, y) <= ols_gmm(βtrue, X, y) + 1e-8
    @test norm(X' * (y - X * βols)) < 1e-8  # Normal equations
    @test ols_gmm(βols, X, y) >= 0

    fit = optimize(b -> ols_gmm(b, X, y), zeros(4), LBFGS(),
                   Optim.Options(iterations=2000, g_tol=1e-7))
    @test Optim.minimum(fit) <= ols_gmm(βtrue, X, y) + 1e-6
    @test isapprox(Optim.minimizer(fit), βols; atol=1e-3, rtol=1e-3)
end

@testset "Q2(a): MLE likelihood, normalization and convexity" begin
    X, y, α = small_logit_fixture()
    N = length(y)
    J = 3
    P = reference_probabilities(α, X)
    @test size(P) == (N, J)
    @test all(P .> 0)
    @test all(isapprox.(vec(sum(P; dims=2)), ones(N); atol=1e-12))

    expected_nll = -sum(log(P[i, y[i]]) for i in 1:N)
    @test isapprox(mlogit_mle(α, X, y), expected_nll; atol=1e-10)
    @test isapprox(mlogit_mle(zeros(length(α)), X, y), N * log(J); atol=1e-10)
    @test mlogit_mle(α, X, y) >= 0
    @test isfinite(mlogit_mle(α, X, y))

    # Duplicate observations must double the negative log likelihood.
    @test isapprox(mlogit_mle(α, vcat(X, X), vcat(y, y)),
                   2 * mlogit_mle(α, X, y); atol=1e-10)

    # Negative multinomial-logit log-likelihood is globally CONVEX.
    αother = -0.6 .* α
    λ = 0.37
    @test mlogit_mle(λ .* α .+ (1 - λ) .* αother, X, y) <=
          λ * mlogit_mle(α, X, y) +
          (1 - λ) * mlogit_mle(αother, X, y) + 1e-10

    # Independent analytical score check against central differences.
    d = choice_indicators(y, J)
    analytical_score = vec(hcat([X' * (P[:, j] .- d[:, j]) for j in 1:J-1]...))
    h = 1e-6
    numerical_score = zeros(length(α))
    for k in eachindex(α)
        aplus = copy(α)
        aminus = copy(α)
        aplus[k] += h
        aminus[k] -= h
        numerical_score[k] = (mlogit_mle(aplus, X, y) -
                              mlogit_mle(aminus, X, y)) / (2 * h)
    end
    @test isapprox(numerical_score, analytical_score; atol=1e-5, rtol=1e-5)
end

@testset "Q2(b,c): GMM moments and overidentified objective" begin
    X, y, α = small_logit_fixture()
    N, K = size(X)
    J = 3
    P = reference_probabilities(α, X)
    d = choice_indicators(y, J)
    g_overid = vec(d .- P)
    @test length(g_overid) == N * J

    # Overidentified GMM: g is N*J long, and W=I; NO N multiplier.
    expected_overid = dot(g_overid, g_overid)
    @test isapprox(mlogit_gmm_overid(α, X, y), expected_overid; atol=1e-10)
    @test mlogit_gmm_overid(α, X, y) >= 0
    @test isfinite(mlogit_gmm_overid(α, X, y))

    # Just-identified GMM: K*(J-1) sample-average moments; N*g'g.
    g_just = Float64[]
    for j in 1:(J-1)
        for k in 1:K
            push!(g_just, mean((d[:, j] .- P[:, j]) .* X[:, k]))
        end
    end
    @test length(g_just) == K * (J - 1)
    @test isapprox(mlogit_gmm(α, X, y), N * dot(g_just, g_just); atol=1e-10)
    @test mlogit_gmm(α, X, y) >= 0
    @test isapprox(mlogit_gmm(zeros(length(α)), X, y),
                   N * dot(vec(X' * (d[:, 1:2] .- 1 / J)) ./ N,
                           vec(X' * (d[:, 1:2] .- 1 / J)) ./ N);
                   atol=1e-10)

    # The negative MLE score equals -N times the GMM moment vector.
    mle_score = vec(hcat([X' * (P[:, j] .- d[:, j]) for j in 1:J-1]...))
    @test isapprox(mle_score, -N .* g_just; atol=1e-10)

    # Duplicating identical data doubles both sample-size-scaled criteria.
    @test isapprox(mlogit_gmm_overid(α, vcat(X, X), vcat(y, y)),
                   2 * expected_overid; atol=1e-9)
    @test isapprox(mlogit_gmm(α, vcat(X, X), vcat(y, y)),
                   2 * mlogit_gmm(α, X, y); atol=1e-9)

    # The professor EXPECTS the random-start and MLE-start GMM
    # optimizations to be compared, not forcibly made equal.
    # This unit test therefore checks the OBJECTIVE FORMULA,
    # not equality of results from different initial conditions.
end

@testset "Q3: inverse-CDF multinomial logit simulation" begin
    Random.seed!(303)
    N = 12_000
    Y, X = sim_logit(N, 4)
    @test length(Y) == N
    @test size(X) == (N, 3)
    @test all(X[:, 1] .== 1.0)
    @test all(isfinite, X)
    @test eltype(Y) <: Integer
    @test all((Y .>= 1) .& (Y .<= 4))
    @test length(unique(Y)) == 4

    β = simulation_coefficients()
    @test size(β) == (3, 4)
    @test β[:, 4] == zeros(3)
    P = reference_probabilities(vec(β[:, 1:3]), X)
    for j in 1:4
        @test abs(mean(Y .== j) - mean(P[:, j])) < 0.025
    end

    # Repeatability under a fixed seed.
    Random.seed!(501)
    Y1, X1 = sim_logit(300, 4)
    Random.seed!(501)
    Y2, X2 = sim_logit(300, 4)
    @test X1 == X2
    @test Y1 == Y2

    # Non-default choice-set branch (random coefficients).
    Random.seed!(502)
    Y5, X5 = sim_logit(2000, 5)
    @test size(X5) == (2000, 3)
    @test all((Y5 .>= 1) .& (Y5 .<= 5))
end

@testset "Q3: Gumbel / Type-I extreme-value simulation" begin
    Random.seed!(404)
    N = 12_000
    Y, X = sim_logit_w_gumbel(N, 4)
    @test length(Y) == N
    @test size(X) == (N, 3)
    @test all(X[:, 1] .== 1.0)
    @test eltype(Y) <: Integer
    @test all((Y .>= 1) .& (Y .<= 4))
    @test length(unique(Y)) == 4
    P = reference_probabilities(vec(simulation_coefficients()[:, 1:3]), X)
    for j in 1:4
        @test abs(mean(Y .== j) - mean(P[:, j])) < 0.025
    end

    Random.seed!(503)
    Y1, X1 = sim_logit_w_gumbel(300, 4)
    Random.seed!(503)
    Y2, X2 = sim_logit_w_gumbel(300, 4)
    @test X1 == X2
    @test Y1 == Y2

    Random.seed!(504)
    Y3, X3 = sim_logit_w_gumbel(1500, 3)
    @test size(X3) == (1500, 3)
    @test all((Y3 .>= 1) .& (Y3 .<= 3))
end

@testset "Q3: two simulation methods imply the same choice frequencies" begin
    Random.seed!(405)
    Ycdf, _ = sim_logit(10_000, 4)
    Random.seed!(406)
    Ygumbel, _ = sim_logit_w_gumbel(10_000, 4)
    for j in 1:4
        @test abs(mean(Ycdf .== j) - mean(Ygumbel .== j)) < 0.03
    end
end

@testset "Q5: SMM moments, scaling, seed and edge conditions" begin
    X, y, α = small_logit_fixture()
    N, K = size(X)
    J = length(α) ÷ K + 1
    D = 4
    bigα = hcat(reshape(α, K, J-1), zeros(K))
    d = choice_indicators(y, J)

    # Independently rebuild simulated frequencies with the same shocks.
    freq = zeros(N, J)
    Random.seed!(1234)
    for draw in 1:D
        ε = rand(Gumbel(0, 1), N, J)
        ys = argmax.(eachrow(X * bigα .+ ε))
        for j in 1:J
            freq[:, j] .+= (ys .== j) .* (1 / D)
        end
    end
    expected_smm = N * sum(abs2, vec(d .- freq))
    actual_smm = mlogit_smm_overid(α, X, y, D)
    @test isapprox(actual_smm, expected_smm; atol=1e-9, rtol=1e-12)
    @test actual_smm >= 0
    @test isfinite(actual_smm)
    @test mlogit_smm_overid(α, X, y, D) == actual_smm
    @test_throws AssertionError mlogit_smm_overid(α, X, y, 0)
    @test_throws AssertionError mlogit_smm_overid(α, X, y, -2)

    # Exact-fit sanity check: D=1 and observed y made with the SAME
    # seeded Gumbel shocks must give an SMM objective of zero.
    Random.seed!(1234)
    ε = rand(Gumbel(0, 1), N, J)
    y_exact = argmax.(eachrow(X * bigα .+ ε))
    @test isapprox(mlogit_smm_overid(α, X, y_exact, 1), 0.0; atol=1e-12)

    # On data simulated with the known coefficients, the true model
    # should fit MUCH better than zero parameters (not exactly zero).
    Random.seed!(505)
    Ysim, Xsim = sim_logit(1000, 4)
    αtrue = vec(simulation_coefficients()[:, 1:3])
    vtrue = mlogit_smm_overid(αtrue, Xsim, Ysim, 12)
    vzero = mlogit_smm_overid(zeros(9), Xsim, Ysim, 12)
    @test isfinite(vtrue) && isfinite(vzero)
    @test vtrue < vzero
end

@testset "Q6: main() is defined without executing it on include" begin
    @test isdefined(@__MODULE__, :main)
    @test hasmethod(main, Tuple{})
    # main() is intentionally NOT called by default: it downloads
    # real data and performs long optimizations, including SMM.
end

# Question 7 also asks us to run the tests. Enable these extra
# integration tests if you need fuller empirical verification.
if get(ENV, "PS7_TEST_SLOW", "0") == "1"
    @testset "SLOW: Q3 parameter recovery with actual MLE optimization" begin
        βtrue = simulation_coefficients()
        αtrue = vec(βtrue[:, 1:3])
        for (seed, simulator) in [(601, sim_logit), (602, sim_logit_w_gumbel)]
            Random.seed!(seed)
            y, X = simulator(20_000, 4)
            αstart = zeros(9)
            fit = optimize(a -> mlogit_mle(a, X, y), αstart,
                           LBFGS(), Optim.Options(iterations=1500,
                                                  g_tol=1e-5,
                                                  show_trace=false))
            @test isfinite(Optim.minimum(fit))
            @test Optim.minimum(fit) < mlogit_mle(αstart, X, y)
            @test maximum(abs.(Optim.minimizer(fit) .- αtrue)) < 0.35
        end
    end
end

if get(ENV, "PS7_TEST_NETWORK", "0") == "1"
    @testset "NETWORK: load_data(url) using the real PS7 CSV" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        @test nrow(df) > 100
        @test size(X) == (nrow(df), 4)
        @test length(y) == nrow(df)
        @test X[:, 1] == ones(nrow(df))
        @test X[:, 2] == df.age
        @test X[:, 3] == Float64.(df.race .== 1)
        @test X[:, 4] == Float64.(df.collgrad .== 1)
        @test isapprox(y, log.(df.wage))
        @test all(isfinite, y)
    end
end

if get(ENV, "PS7_TEST_STRESS", "0") == "1"
    @testset "STRESS: numerical stability at extreme parameter values" begin
        X, y, α = small_logit_fixture()
        extreme = 1000 .* α
        # The current source uses exp.(utilities) without subtracting
        # per-row maxima in MLE and overidentified GMM. These may FAIL
        # for extreme utilities: these tests diagnose that limitation.
        @test isfinite(mlogit_mle(extreme, X, y))
        @test isfinite(mlogit_gmm_overid(extreme, X, y))
        # The just-identified GMM uses stable, centered softmax.
        @test isfinite(mlogit_gmm(extreme, X, y))
    end
end

if get(ENV, "PS7_TEST_FULL", "0") == "1"
    @testset "FULL integration: run original main() (long)" begin
        results = main()
        @test length(results) == 6
        @test all(isfinite, Optim.minimizer(results[1]))
        @test all(isfinite, Optim.minimizer(results[2]))
        @test all(isfinite, Optim.minimizer(results[3]))
        @test all(isfinite, Optim.minimizer(results[4]))
        @test all(isfinite, Optim.minimizer(results[5]))
        @test all(isfinite, Optim.minimizer(results[6]))
        @test Optim.minimum(results[3]) >= 0
        @test Optim.minimum(results[4]) >= 0
        @test isfinite(Optim.minimum(results[6]))
    end
end

println("\nAll PS7 unit tests completed.")
