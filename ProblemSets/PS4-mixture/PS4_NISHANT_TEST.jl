using Test
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using Distributions
using ForwardDiff

# ============================================================
# Read required functions
# ============================================================

include("lgwt.jl")
include("PS4_NISHANT_SOURCE.jl")


println("\n==============================================")
println("      COMPREHENSIVE UNIT TESTS - PS4")
println("==============================================\n")


# ============================================================
# Load data once for use throughout tests
# ============================================================

df, X, Z, y = load_data()

N = size(X, 1)
K = size(X, 2)
J = length(unique(y))


# ============================================================
# TEST SET 1: DATA LOADING
# ============================================================

@testset "1. Data Loading" begin

    # Data should not be empty
    @test nrow(df) > 0

    # X should contain age, white, collgrad
    @test size(X, 2) == 3

    # Z should contain expected wages for 8 occupations
    @test size(Z, 2) == 8

    # All variables should contain same number of observations
    @test size(X, 1) == size(Z, 1)
    @test size(X, 1) == length(y)

    # There should be 8 occupational alternatives
    @test length(unique(y)) == 8

    # Occupation codes should be 1,...,8
    @test sort(unique(y)) == collect(1:8)

    # Check data contain finite values
    @test all(isfinite, X)
    @test all(isfinite, Z)

    println("✓ Data loading and dimensions are correct.")
end


# ============================================================
# TEST SET 2: MULTINOMIAL LOGIT - BASIC FUNCTIONALITY
# ============================================================

@testset "2. Multinomial Logit Basic Functionality" begin

    # 21 alpha coefficients + 1 gamma
    theta = [zeros(K * (J - 1)); 0.1]

    ll = mlogit_with_Z(theta, X, Z, y)

    # Negative log likelihood should return a scalar
    @test ll isa Real

    # It should be finite
    @test isfinite(ll)

    # Negative log likelihood should be positive
    @test ll > 0

    println("✓ Multinomial logit returns a valid objective value.")
end


# ============================================================
# TEST SET 3: MULTINOMIAL LOGIT - KNOWN SPECIAL CASE
# ============================================================

@testset "3. Multinomial Logit Uniform Probability Test" begin

    # If every parameter equals zero:
    #
    # V_ij = 0 for every alternative
    #
    # Therefore:
    # P_ij = 1/J
    #
    # Negative log likelihood should equal:
    #
    # -sum(log(1/J)) = N*log(J)

    theta_zero = zeros(K * (J - 1) + 1)

    ll_zero = mlogit_with_Z(theta_zero, X, Z, y)

    theoretical_ll = N * log(J)

    @test isapprox(
        ll_zero,
        theoretical_ll;
        rtol = 1e-10,
        atol = 1e-8
    )

    println("✓ Zero coefficients produce uniform choice probabilities.")
    println("  Computed NLL:    ", ll_zero)
    println("  Theoretical NLL: ", theoretical_ll)
end


# ============================================================
# TEST SET 4: GAMMA = 0 IMPLIES Z SHOULD NOT MATTER
# ============================================================

@testset "4. Alternative-Specific Covariate Test" begin

    Random.seed!(1234)

    alpha = randn(K * (J - 1))

    # Set gamma = 0
    theta_gamma_zero = [alpha; 0.0]

    ll_original =
        mlogit_with_Z(theta_gamma_zero, X, Z, y)

    # Completely change Z
    Z_fake = randn(size(Z))

    ll_fakeZ =
        mlogit_with_Z(theta_gamma_zero, X, Z_fake, y)

    # With gamma = 0, Z drops out of utility,
    # so likelihood must be identical
    @test isapprox(
        ll_original,
        ll_fakeZ;
        rtol = 1e-12,
        atol = 1e-10
    )

    println("✓ When gamma = 0, changing Z does not affect likelihood.")
end


# ============================================================
# TEST SET 5: AUTOMATIC DIFFERENTIATION
# ============================================================

@testset "5. Automatic Differentiation" begin

    # Use parameters close to zero to avoid overflow
    theta = [fill(0.01, K * (J - 1)); 0.1]

    # ForwardDiff should be able to calculate gradient
    grad = ForwardDiff.gradient(
        t -> mlogit_with_Z(t, X, Z, y),
        theta
    )

    @test length(grad) == length(theta)
    @test all(isfinite, grad)

    # Hessian should also be computable
    H = ForwardDiff.hessian(
        t -> mlogit_with_Z(t, X, Z, y),
        theta
    )

    @test size(H) == (length(theta), length(theta))
    @test all(isfinite, H)

    # Numerical Hessian should be symmetric
    @test isapprox(
        H,
        H';
        rtol = 1e-8,
        atol = 1e-8
    )

    println("✓ ForwardDiff gradient and Hessian work correctly.")
end


# ============================================================
# TEST SET 6: QUESTION 3a - QUADRATURE NODES AND WEIGHTS
# ============================================================

@testset "6. Quadrature Nodes and Weights" begin

    nodes, weights = lgwt(7, -4, 4)

    # Correct number of points
    @test length(nodes) == 7
    @test length(weights) == 7

    # Nodes must lie within integration bounds
    @test all((-4 .<= nodes) .& (nodes .<= 4))

    # Gauss-Legendre weights should be positive
    @test all(weights .> 0)

    # Weights over [a,b] should sum to b-a = 8
    @test isapprox(sum(weights), 8.0; atol=1e-10)

    # Nodes should be symmetric around zero
    @test isapprox(
        nodes,
        -reverse(nodes);
        atol = 1e-10
    )

    # Weights should also be symmetric
    @test isapprox(
        weights,
        reverse(weights);
        atol = 1e-10
    )

    println("✓ Quadrature nodes and weights have correct properties.")
end


# ============================================================
# TEST SET 7: QUESTION 3a - STANDARD NORMAL INTEGRALS
# ============================================================

@testset "7. Standard Normal Quadrature" begin

    d = Normal(0, 1)

    nodes, weights = lgwt(7, -4, 4)

    density_integral =
        sum(weights .* pdf.(d, nodes))

    mean_integral =
        sum(weights .* nodes .* pdf.(d, nodes))

    # Integral of density approximately equals one
    @test isapprox(
        density_integral,
        1.0;
        atol = 0.01
    )

    # Mean of symmetric N(0,1) should equal zero
    @test isapprox(
        mean_integral,
        0.0;
        atol = 1e-10
    )

    println("✓ ∫φ(x)dx ≈ 1")
    println("✓ ∫xφ(x)dx ≈ 0")
end


# ============================================================
# TEST SET 8: QUESTION 3b - VARIANCE QUADRATURE
# ============================================================

@testset "8. Variance Quadrature" begin

    sigma = 2.0
    d = Normal(0, sigma)

    # ----------------------
    # 7 quadrature points
    # ----------------------

    nodes7, weights7 =
        lgwt(7, -5*sigma, 5*sigma)

    variance7 =
        sum(
            weights7 .*
            nodes7.^2 .*
            pdf.(d, nodes7)
        )

    # ----------------------
    # 10 quadrature points
    # ----------------------

    nodes10, weights10 =
        lgwt(10, -5*sigma, 5*sigma)

    variance10 =
        sum(
            weights10 .*
            nodes10.^2 .*
            pdf.(d, nodes10)
        )

    true_variance = sigma^2

    # Basic validity
    @test isfinite(variance7)
    @test isfinite(variance10)

    @test variance7 > 0
    @test variance10 > 0

    # 10-point approximation should be closer to 4
    @test abs(variance10 - true_variance) <
          abs(variance7 - true_variance)

    # 10-point result should be close to theoretical variance
    @test isapprox(
        variance10,
        true_variance;
        atol = 0.1
    )

    println("✓ True variance:     ", true_variance)
    println("✓ 7-point estimate:  ", variance7)
    println("✓ 10-point estimate: ", variance10)
    println("✓ 10-point quadrature improves the approximation.")
end


# ============================================================
# TEST SET 9: QUESTION 3 FUNCTIONS RUN WITHOUT ERROR
# ============================================================

@testset "9. Question 3 Functions Execute" begin

    # These functions mainly print their answers,
    # so we verify they run successfully.

    @test practice_quadrature() === nothing

    @test variance_quadrature() === nothing

    Random.seed!(1234)
    @test practice_monte_carlo() === nothing

    println("✓ Question 3 functions execute without error.")
end


# ============================================================
# TEST SET 10: MONTE CARLO INTEGRATION
# ============================================================

@testset "10. Monte Carlo Integration" begin

    Random.seed!(1234)

    sigma = 2.0
    d = Normal(0, sigma)

    a = -5*sigma
    b = 5*sigma

    # Large enough for a stable unit test,
    # but smaller than 1,000,000 to keep tests fast
    D = 200_000

    draws =
        rand(D) .* (b - a) .+ a

    # ------------------------------------------------
    # Integral of density
    # ------------------------------------------------

    density_mc =
        (b-a) * mean(pdf.(d, draws))

    # ------------------------------------------------
    # Mean
    # ------------------------------------------------

    mean_mc =
        (b-a) *
        mean(draws .* pdf.(d, draws))

    # ------------------------------------------------
    # Variance
    # ------------------------------------------------

    variance_mc =
        (b-a) *
        mean(
            draws.^2 .* pdf.(d, draws)
        )

    @test isapprox(
        density_mc,
        1.0;
        atol = 0.03
    )

    @test isapprox(
        mean_mc,
        0.0;
        atol = 0.08
    )

    @test isapprox(
        variance_mc,
        4.0;
        atol = 0.15
    )

    println("✓ Monte Carlo density integral ≈ 1")
    println("✓ Monte Carlo mean ≈ 0")
    println("✓ Monte Carlo variance ≈ 4")
end


# ============================================================
# TEST SET 11: MORE DRAWS GENERALLY REDUCE MC ERROR
# ============================================================

@testset "11. Monte Carlo Draw Comparison" begin

    sigma = 2.0
    d = Normal(0, sigma)

    a = -5*sigma
    b = 5*sigma

    # Use the same seed so test is reproducible
    Random.seed!(2026)

    # -------------------
    # D = 1,000
    # -------------------

    draws_small =
        rand(1_000) .* (b-a) .+ a

    variance_small =
        (b-a) *
        mean(
            draws_small.^2 .*
            pdf.(d, draws_small)
        )

    # -------------------
    # D = 1,000,000
    # -------------------

    Random.seed!(2026)

    draws_large =
        rand(1_000_000) .* (b-a) .+ a

    variance_large =
        (b-a) *
        mean(
            draws_large.^2 .*
            pdf.(d, draws_large)
        )

    error_small =
        abs(variance_small - 4.0)

    error_large =
        abs(variance_large - 4.0)

    # For this fixed seed, the large simulation should
    # give the closer approximation
    @test error_large < error_small

    println("✓ Error with 1,000 draws:     ", error_small)
    println("✓ Error with 1,000,000 draws: ", error_large)
end


# ============================================================
# TEST SET 12: QUESTION 4 - MIXED LOGIT BASIC TEST
# ============================================================

@testset "12. Mixed Logit Quadrature Basic Functionality" begin

    nodes, weights = lgwt(7, -4, 4)

    # 21 alpha coefficients
    # + mu_gamma
    # + sigma_gamma
    theta_mixed =
        [zeros(K * (J - 1)); 0.1; 1.0]

    ll_mixed =
        mixed_logit_quad(
            theta_mixed,
            X,
            Z,
            y,
            nodes,
            weights
        )

    @test ll_mixed isa Real
    @test isfinite(ll_mixed)
    @test ll_mixed > 0

    println("✓ Mixed-logit quadrature likelihood returns a valid value.")
end


# ============================================================
# TEST SET 13: MIXED LOGIT WITH SIGMA = 0
# ============================================================

@testset "13. Mixed Logit Degenerate Distribution Test" begin

    nodes, weights = lgwt(7, -4, 4)

    mu_gamma = 0.1

    # If sigma_gamma = 0, every quadrature node gives:
    #
    # gamma_r = mu_gamma
    #
    # Therefore the mixed logit should collapse toward
    # an ordinary multinomial logit with gamma = mu_gamma,
    # apart from the tiny truncation error from integrating
    # the normal density only over [-4,4].

    theta_logit =
        [zeros(K * (J - 1)); mu_gamma]

    theta_mixed =
        [zeros(K * (J - 1)); mu_gamma; 0.0]

    ll_logit =
        mlogit_with_Z(
            theta_logit,
            X,
            Z,
            y
        )

    ll_mixed =
        mixed_logit_quad(
            theta_mixed,
            X,
            Z,
            y,
            nodes,
            weights
        )

    # Quadrature only integrates N(0,1) over [-4,4].
    # Its approximated total probability mass is:
    mass =
        sum(
            weights .*
            pdf.(Normal(0,1), nodes)
        )

    # P_mixed = P_logit * mass
    # Therefore:
    #
    # NLL_mixed = NLL_logit - N*log(mass)

    expected_mixed =
        ll_logit - N * log(mass)

    @test isapprox(
        ll_mixed,
        expected_mixed;
        rtol = 1e-8,
        atol = 1e-6
    )

    println("✓ Mixed logit collapses correctly when sigma_gamma = 0.")
end


# ============================================================
# TEST SET 14: MIXED LOGIT QUADRATURE NODE TRANSFORMATION
# ============================================================

@testset "14. Random-Coefficient Transformation" begin

    nodes, weights = lgwt(7, -4, 4)

    mu_gamma = 1.0
    sigma_gamma = 0.5

    gamma_nodes =
        mu_gamma .+
        sigma_gamma .* nodes

    # Since quadrature nodes are symmetric around zero,
    # transformed gamma values should be symmetric around mu.
    @test isapprox(
        mean(
            [
                gamma_nodes[i] +
                gamma_nodes[end-i+1]
                for i in 1:length(gamma_nodes)
            ]
        ) / 2,
        mu_gamma;
        atol = 1e-10
    )

    # Center quadrature node should transform into mu
    @test isapprox(
        gamma_nodes[4],
        mu_gamma;
        atol = 1e-10
    )

    println("✓ gamma_r = mu_gamma + sigma_gamma*node_r works correctly.")
end


# ============================================================
# IMPORTANT: DO NOT RUN QUESTION 4 OPTIMIZATION
# ============================================================

@testset "15. Question 4 Optimization Safety" begin

    # The assignment explicitly says to modify but NOT RUN
    # the mixed-logit optimization.
    #
    # Therefore we intentionally test only the likelihood
    # function above and do not call:
    #
    # optimize_mixed_logit_quad(X, Z, y)

    @test true

    println("✓ Mixed-logit optimization was intentionally NOT executed.")
end


# ============================================================
# FINAL MESSAGE
# ============================================================

println("\n==============================================")
println("       ALL COMPREHENSIVE TESTS PASSED")
println("==============================================")
println("Questions 1-4 functions/components verified.")
println("Mixed-logit optimization was NOT executed.")