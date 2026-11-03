using Test
using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

include("PS8_NISHANT_SOURCE.jl")
include("lgwt.jl")

#==================================================================================
# Question 7: Unit Tests
==================================================================================#

# Share local fixtures while letting each test set print its own summary.
let
    rng = MersenneTwister(8)
    n = 100
    covariates = [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]
    scores = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    df = DataFrame(randn(rng, n, 6), covariates)
    latent = randn(rng, n)
    for (j, name) in enumerate(scores)
        df[!, name] = (0.5 + j / 10) .* latent + randn(rng, n)
    end
    coefficients = collect(0.1:0.1:0.7)
    design = hcat(ones(n), Matrix(df[:, covariates]))
    df.logwage = design * coefficients

    @testset "Regressions" begin
        base = estimate_base_regression(df)
        @test coef(base) ≈ coefficients
        full = estimate_full_regression(df)
        @test coef(full)[1:7] ≈ coefficients
        @test coef(full)[8:end] ≈ zeros(6) atol=1e-12
        for (estimate, column) in ((estimate_pca_regression, :PC),
                                   (estimate_factor_regression, :FA))
            data = copy(df)
            model = estimate(data)
            @test length(coef(model)) == 8
            @test predict(model) ≈ df.logwage
            @test length(data[!, column]) == n
            @test all(isfinite, data[!, column])
        end
    end

    @testset "Correlations and matrix preparation" begin
        correlations = compute_asvab_correlations(df)
        @test correlations isa DataFrame
        @test size(correlations) == (6, 6)
        @test names(correlations) == string.(scores)
        @test Matrix(correlations) ≈ cor(Matrix(df[:, scores]))
        @test diag(Matrix(correlations)) ≈ ones(6)
        X, y, Xfac, measurements = prepare_factor_matrices(df)
        @test size(X) == (n, 7)
        @test X == hcat(Matrix(df[:, covariates]), ones(n))
        @test y == df.logwage
        @test size(Xfac) == (n, 4)
        @test Xfac == hcat(Matrix(df[:, covariates[1:3]]), ones(n))
        @test measurements == Matrix(df[:, scores])
    end

    @testset "Likelihood and parameter ordering" begin
        X = [1.0 0.2; 1.0 -0.3; 1.0 0.5]
        Xfac = ones(3, 1)
        measurements = [0.2 -0.4; 0.5 0.3; -0.1 0.7]
        y = [0.1, -0.2, 0.4]
        α = reshape([0.1, -0.2], 1, 2)
        β = [0.2, -0.1]
        γ = [0.4, 0.6]
        δ = 0.3
        σ = [0.8, 1.1, 0.9]
        θ = vcat(vec(α), β, γ, δ, σ)
        nodes, weights = lgwt(20, -5, 5)
        expected = -sum(1:3) do i
            log(sum(eachindex(nodes)) do r
                ξ = nodes[r]
                weights[r] * pdf(Normal(), ξ) *
                    pdf(Normal(dot(X[i, :], β) + δ * ξ, σ[3]), y[i]) *
                    prod(pdf(Normal(dot(Xfac[i, :], α[:, j]) + γ[j] * ξ, σ[j]),
                             measurements[i, j]) for j in 1:2)
            end)
        end
        objective = θ -> factor_model(θ, X, Xfac, measurements, y, 20)
        @test isfinite(objective(θ))
        @test objective(θ) ≈ expected
        @test all(isfinite, ForwardDiff.gradient(objective, θ))

        # With zero loadings and fixed unit scales, the wage intercept MLE is mean(y).
        fixed = vcat(zeros(7), ones(3))
        result = Optim.optimize([-1.0], Optim.BFGS()) do intercept
            parameters = copy(fixed)
            parameters[3] = intercept[1]
            objective(parameters)
        end
        @test Optim.converged(result)
        @test Optim.minimizer(result)[1] ≈ mean(y) atol=1e-6
    end

    @testset "Result formatting" begin
        θ = collect(1.0:45.0)
        se = fill(0.1, 45)
        results = format_results(θ, se, -10.0, zeros(2, 6))
        @test size(results) == (45, 4)
        @test names(results) == ["equation", "variable", "coefficient", "std_error"]
        @test results.coefficient == θ
        @test results.std_error == se
        @test results.equation[1:4] == fill("asvabAR", 4)
        @test results.variable[1:4] == ["black", "hispanic", "female", "constant"]
        @test results.equation[25:31] == fill("logwage", 7)
        @test results.variable[32:38] == fill("factor_loading", 7)
        @test results.variable[39:45] == fill("sigma", 7)
    end

    @testset "CSV loading over local HTTP" begin
        csv = "black,logwage,schoolt\n0,1.25,12\n1,2.5,16\n"
        server = HTTP.serve!(_ -> HTTP.Response(200, csv), "127.0.0.1", 0)
        try
            data = load_data("http://127.0.0.1:$(HTTP.port(server))/fixture.csv")
            @test data isa DataFrame
            @test names(data) == ["black", "logwage", "schoolt"]
            @test data == DataFrame(black=[0, 1], logwage=[1.25, 2.5], schoolt=[12, 16])
        finally
            close(server)
        end
    end

    @testset "Column selection and regression signals" begin
        original = copy(df)
        reordered = df[:, reverse(names(df))]
        reordered.unrelated = randn(rng, n)
        @test compute_asvab_correlations(reordered) == compute_asvab_correlations(df)
        @test prepare_factor_matrices(reordered) == prepare_factor_matrices(df)
        @test coef(estimate_base_regression(reordered)) ≈ coefficients
        @test df == original
        @test_throws ArgumentError compute_asvab_correlations(select(df, Not(:asvabAR)))
        @test_throws ArgumentError prepare_factor_matrices(select(df, Not(:schoolt)))

        data = copy(df)
        score_coefficients = collect(0.2:0.2:1.2)
        data.logwage += Matrix(data[:, scores]) * score_coefficients
        @test coef(estimate_full_regression(data)) ≈ vcat(coefficients, score_coefficients)
        for (estimate, column) in ((estimate_pca_regression, :PC),
                                   (estimate_factor_regression, :FA))
            data = copy(df)
            estimate(data)
            signal = copy(data[!, column])
            data.logwage += 0.75 .* signal
            model = estimate(data)
            @test coef(model)[end] ≈ 0.75
            @test predict(model) ≈ data.logwage
            @test Matrix(data[:, scores]) == Matrix(df[:, scores])
        end
        data = copy(df)
        estimate_pca_regression(data)
        after_pca = estimate_factor_regression(data)
        @test predict(after_pca) ≈ predict(estimate_factor_regression(copy(df)))
    end

    @testset "Quadrature accuracy" begin
        nodes, weights = lgwt(20, -5, 5)
        @test length(nodes) == length(weights) == 20
        @test all(-5 .< nodes .< 5)
        @test all(weights .> 0)
        @test sum(weights) ≈ 10
        @test sum(weights .* nodes) ≈ 0 atol=1e-13
        @test sum(weights .* nodes.^2) ≈ 250 / 3
    end

    @testset "Analytical Gaussian likelihood and derivatives" begin
        # Integrating a Gaussian factor gives covariance λλ′ + Diagonal(σ²).
        for (N, K, L, J) in ((4, 2, 1, 1), (5, 3, 2, 3), (3, 7, 4, 6))
            X = randn(rng, N, K)
            Xfac = randn(rng, N, L)
            α = 0.1 .* randn(rng, L, J)
            β = 0.1 .* randn(rng, K)
            λ = 0.2 .+ 0.1 .* rand(rng, J + 1)
            σ = 0.8 .+ 0.2 .* rand(rng, J + 1)
            measurements = Xfac * α + 0.3 .* randn(rng, N, J)
            y = X * β + 0.3 .* randn(rng, N)
            θ = vcat(vec(α), β, λ, σ)
            objective = t -> factor_model(t, X, Xfac, measurements, y, 50)
            residuals = hcat(measurements - Xfac * α, y - X * β)
            gaussian = MvNormal(zeros(J + 1), λ * λ' + Diagonal(σ.^2))
            analytical = -sum(logpdf(gaussian, vec(residuals[i, :])) for i in 1:N)
            @test objective(θ) ≈ analytical atol=1e-5

            # The implementation truncates the latent normal integral to [-5, 5].
            independent = copy(θ)
            loading_indices = (L * J + K + 1):(L * J + K + J + 1)
            independent[loading_indices] .= 0
            expected = -sum(logpdf(Normal(0, σ[j]), residuals[i, j])
                            for i in 1:N, j in 1:(J + 1)) -
                       N * log(cdf(Normal(), 5) - cdf(Normal(), -5))
            @test objective(independent) ≈ expected atol=1e-10
            flipped = copy(θ)
            flipped[loading_indices] .*= -1
            @test objective(flipped) ≈ objective(θ)
            @test factor_model(θ, X, Xfac, measurements, y, 40) ≈ objective(θ) atol=1e-9
            order = reverse(1:N)
            @test factor_model(θ, X[order, :], Xfac[order, :],
                               measurements[order, :], y[order], 50) ≈ objective(θ)
            @test factor_model(θ, vcat(X, X), vcat(Xfac, Xfac),
                               vcat(measurements, measurements), vcat(y, y), 50) ≈ 2objective(θ)

            step = 1e-5
            numerical_gradient = map(eachindex(θ)) do j
                plus, minus = copy(θ), copy(θ)
                plus[j] += step
                minus[j] -= step
                (objective(plus) - objective(minus)) / (2step)
            end
            @test ForwardDiff.gradient(objective, θ) ≈ numerical_gradient atol=1e-7 rtol=1e-6
            H = ForwardDiff.hessian(objective, θ)
            @test all(isfinite, H)
            @test H ≈ H'
        end
    end

    @testset "Full estimation integration" begin
        # A separate noisy sample avoids the zero residual variance of the OLS fixture.
        simulation_rng = MersenneTwister(123)
        N = 240
        data = DataFrame(randn(simulation_rng, N, 6), covariates)
        factor = randn(simulation_rng, N)
        for name in scores
            data[!, name] = 0.7 .* factor + randn(simulation_rng, N)
        end
        data.logwage = 0.4 .* data.schoolt + 0.7 .* factor + randn(simulation_rng, N)
        X, y, Xfac, measurements = prepare_factor_matrices(data)
        initial = vcat(vec(Xfac \ measurements), X \ y, fill(0.7, 7), ones(7))
        original_initial = copy(initial)
        θhat, se, loglike = run_estimation(data, initial)
        @test length(θhat) == length(se) == 45
        @test all(isfinite, θhat)
        @test all(isfinite, se)
        @test all(se .> 0)
        @test all(θhat[39:45] .> 0)
        @test initial == original_initial
        objective = t -> factor_model(t, X, Xfac, measurements, y, 50)
        @test loglike ≈ -objective(θhat)
        @test loglike >= -objective(initial) - 1e-8
        @test norm(ForwardDiff.gradient(objective, θhat), Inf) < 1e-4
        H = ForwardDiff.hessian(objective, θhat)
        @test isposdef(Symmetric(H))
        @test se ≈ sqrt.(diag(inv(H)))
        @test size(format_results(θhat, se, loglike, measurements)) == (45, 4)
    end

end

println("\nAll PS8 unit tests completed.")
