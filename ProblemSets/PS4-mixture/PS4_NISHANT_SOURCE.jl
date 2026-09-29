#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------

function mlogit_with_Z(theta, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements  
    # gamma is the coefficient on Z
    alpha = theta[1:end-1]  # first 21 elements
    gamma = theta[end]      # last element
    
    K = size(X, 2)  # number of covariates in X (3)
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # Compute choice probabilities
    # Hint: P_ij = exp(X_i*beta_j + gamma*(Z_ij - Z_iJ)) / denominator
    # where denominator sums over all choices
    
    # Initialize probability matrix  
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # compute numerator for each choice j
    for j = 1:J
        num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
    end
    
    # compute denominator (sum of numerators)
    dem = sum(num, dims=2)
    
    # compute probabilities
    P = num ./ dem
    
    # compute negative log-likelihood
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end

#---------------------------------------------------
# Question 3a: Quadrature Practice
#---------------------------------------------------

function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")
    
    # Define standard normal distribution
    d = Normal(0, 1)
    
    # Get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7, -4, 4)
    
    integral_density = sum(weights[i] * pdf(d, nodes[i]) for i in eachindex(nodes, weights))
    println("∫φ(x)dx = $integral_density (should be ≈ 1)")
    
    expectation = sum(weights[i] * nodes[i] * pdf(d, nodes[i]) for i in eachindex(nodes, weights))
    @assert isapprox(expectation, 0; atol=1e-12) "Quadrature expectation should be approximately zero"
    println("∫xφ(x)dx = $expectation (should be ≈ 0)")
end

#---------------------------------------------------
# Question 3b: More Quadrature Practice
#---------------------------------------------------

function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")
    
    # Define N(0,2) distribution
    d = Normal(0, 2)
    σ = 2
    
    # Use quadrature to compute ∫x²f(x)dx with 7 points
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    
    # Use quadrature to compute ∫x²f(x)dx with 10 points
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)  
    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    
    println("Variance with 7 quadrature points: ", variance_7pts)
    println("Variance with 10 quadrature points: ", variance_10pts)
    println("True variance: $(σ^2)")
    
    # Comment on approximation quality
    println(""" 
    With 7 quadrature points, the estimate is 3.2655, which is noticeably
    below the true variance of 4. With 10 points, the estimate is 4.0390,
    which is very close to 4. Therefore, the 10-point quadrature provides
    a much better approximation.
    """)
    

end

#---------------------------------------------------
# Question 3c: Monte Carlo Practice  
#---------------------------------------------------

function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")
    
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    
    function mc_integrate(f, a, b, D)
        # ∫f(x)dx ≈ (b-a) * (1/D) * Σf(X_i) where X_i ~ U[a,b]
        draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        return (b - a) * mean(f.(draws))
    end
    
    # Test with different numbers of draws
    for D in [1000, 1000000]
        println("\nWith D = $D draws:")
        
        # Variance: ∫x²f(x)dx  
        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        println("MC Variance: ", variance_mc, " (true: $(σ^2))")
        
        # Mean: ∫xf(x)dx
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)  
        println("MC Mean: ", mean_mc, " (true: 0)")
        
        # Density integral: ∫f(x)dx
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Density integral: ", density_mc, " (true: 1)")
    end

    println(""" 
    With D = 1,000, there is noticeable simulation error.
    With D = 1,000,000, the estimates are much closer to the true values 4, 0, and 1.
    Thus, increasing the number of Monte Carlo draws substantially improves the approximation.
    """)
end

#---------------------------------------------------
# Question 4: Mixed Logit with Quadrature (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    # Extract parameters
    # theta = [alpha1, ..., alpha21, mu_gamma, sigma_gamma]
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]  # coefficients on X
    mu_gamma = theta[end-1]     # mean of gamma distribution
    sigma_gamma = theta[end]    # std dev of gamma distribution
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha 
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # Implement mixed logit with quadrature
    # This involves integrating over the distribution of gamma
    
    # Initialize integrated probabilities
    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # For each quadrature point r:
    # 1. Transform node: gamma_r = mu_gamma + sigma_gamma * nodes[r]
    # 2. Compute choice probabilities for this gamma_r (like regular logit)
    # 3. Weight by quadrature weight and normal density
    # 4. Add to integrated probabilities
    
    for r in eachindex(nodes)
        gamma_r = mu_gamma + sigma_gamma * nodes[r]
        
        # Compute probabilities for this gamma_r
        num_r = zeros(T, N, J)
        for j = 1:J
            num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        end
        dem_r = sum(num_r, dims=2)
        P_r = num_r ./ dem_r
        
        # Weight and add to integrated probabilities
        density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        P_integrated .+= P_r * density_weight
    end
    
    # Compute log-likelihood  
    loglike = -sum(bigY .* log.(P_integrated))
    
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    
    # Starting values: K*(J-1) alphas + 1 gamma
    startvals = [2*rand(K*(J-1)).-1; 0.1]
    
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y),
                             startvals, autodiff = Optim.ADTypes.AutoForwardDiff())
    
    result = optimize(td, startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
        
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, result.minimizer)
    result_se = sqrt.(diag(inv(H)))
    return result.minimizer, result_se
end

function optimize_mixed_logit_quad(X, Z, y)
    K = size(X, 2)  
    J = length(unique(y))
    
    # Get quadrature nodes and weights
    nodes, weights = lgwt(7, -4, 4)
    
    # Starting values: K*(J-1) alphas + mu_gamma + sigma_gamma
    # Use regular logit estimates as starting values for alpha and gamma
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]  # last element is sigma_gamma
    
    # Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
                     startvals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
                     autodiff = :forward)
    
    println("Mixed logit quadrature optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end


#---------------------------------------------------
# Question 6: Main Function
#---------------------------------------------------

function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    
    # Load data
    df, X, Z, y = load_data()
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # Question 1: Estimate multinomial logit
    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle, se_hat = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("Standard errors: ", se_hat)
    alpha_hat = theta_hat_mle[1:end-1]
    gamma_hat = theta_hat_mle[end]
    println("α̂ = ", alpha_hat)
    println("γ̂ = ", gamma_hat)
    
    # Question 2: Interpret gamma
    println("\n=== QUESTION 2: INTERPRETATION ===")
    println(""" In PS3, gamma hat = -0.09, which implied that a higher expected wage reduced utility. 
    This was counterintuitive and suggested possible model misspecification or omitted job characteristics correlated with wages.
    In PS4, gamma hat = $(round(gamma_hat, digits=4)), which makes more economic sense because a
    higher expected log wage now increases the utility of choosing that
    occupation, holding everything else constant. """)
    

    # Question 3: Practice with quadrature and Monte Carlo
    practice_quadrature()
    variance_quadrature()
    practice_monte_carlo()
    println(""" Question 3d:
    Quadrature and Monte Carlo both approximate an integral using a weighted
    sum of function evaluations. In quadrature, the nodes and weights are
    chosen deterministically. In Monte Carlo integration, the nodes are
    random draws from U[a,b] and each receives the same weight (b-a)/D.
    """)
    println("\n=== ALL ANALYSES COMPLETE ===")

    return nothing
    
    # Question 4: Mixed logit with quadrature (setup only)
    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP) ===")
    optimize_mixed_logit_quad(X, Z, y)
    
    
end

println("Starter code loaded successfully!")
println("Remember to:")
println("1. Fill in all TODO sections")
println("2. Test functions step by step")  
println("3. Don't run mixed logit estimations (too computationally intensive)")
println("4. Use automatic differentiation in optimization")
