################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :log_wage = log.(:wage))
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.log_wage
    # could also do:
    # y = log.(df.wage)
    return df, X, y
end


"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS via GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression.
Uses identity weighting matrix.

Mathematical form:
    J(β) = g'Wg where W = I and g = (y - Xβ)

Arguments:
- β: coefficient vector
- X: N×K design matrix
- y: N×1 outcome vector

Returns: scalar objective function value
"""
function ols_gmm(β, X, y)
    # Compute predicted values ŷ = Xβ
    ŷ = X*β
    
    # Compute residuals g = y - ŷ
    g = y - ŷ
    
    # Compute objective function J = g'Ig
    # Hint: Use the identity matrix or just g'*g
    J = g⋅g # g \cdot g
    # I prefer \cdot because it will return a scalar;
    # g'*g will return a 1x1 matrix, which I then have to return J[1]
    
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(α, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- α: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(alpha, X, y)
    N = size(X,1) # no. of obs
    K = size(X,2) # no. of covariates
    J = length(alpha) ÷ K + 1 # number of choice alternatives
    
    # unslice the parameter vector
    alpha_mat = [reshape(alpha, K, J-1) zeros(K, 1)] # add a column of zeros for the base category
    
    # make d matrix
    d = zeros(N, J)
    for j = 1:J
        d[:,j] .= y .== j
    end
    
    # make p matrix
    p = zeros(N, J)
    for j = 1:J
        p[:,j] .= exp.(X * alpha_mat[:, j])./sum(exp.(X * alpha_mat); dims=2)
    end
    
    # log-likelihood expression
    loglike = -sum(d .* log.(p))
    
    return loglike
end

"""
    mlogit_gmm(α, X, y)

GMM objective function for multinomial logit using just-identified moments.

Moment conditions:
    E[X_i(d_ij - P_ij(α))] = 0 for j = 1,...,J-1

This gives K*(J-1) moments for K*(J-1) parameters.
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1  # α has K*(J-1) elements; counts alternatives even if some are never chosen
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = (y .== j)
    end

    # Reshape coefficients
    bigα = hcat(reshape(α, K, J-1), zeros(K))

    # Compute choice probabilities (using stable softmax)
    V = X * bigα
    P = exp.(V .- maximum(V; dims=2))
    P ./= sum(P; dims=2)

    # Compute moment vector g of dimension K*(J-1)
    # Each element is: mean((d_ij - P_ij) * X_ik)
    g = zeros((J-1)*K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:,j] .- P[:,j]) .* X[:,k])
        end
    end

    # Compute objective function obj = N * g'g
    obj = N * g' * g

    return obj
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1 # α has K*(J-1) elements
    N = length(y)
    
    # Create choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Reshape and compute probabilities
    bigα = [reshape(α, K, J-1) zeros(K, 1)] # add a column of zeros for the base category
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(X * bigα[:, j])./sum(exp.(X * bigα); dims=2)
    end
    
    # Stack moments as vector: g = vec(d - P)
    # This creates an N*J dimensional vector
    # NOTE: THIS IS NOT -SUM(D .* LOG(P)) AS IN MLE
    g = bigY[:] .- P[:]
    
    # Compute objective obj = g'Wg with W = I
    obj = g⋅g # g \cdot g
    
    return obj 
end


################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################

"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF method.

Steps:
1. Generate X ~ N(μ, Σ)
2. Set coefficient matrix β (K×J)
3. Compute choice probabilities P_ij
4. Draw uniform random variable ε_i
5. Assign y_i based on which cumulative probability bracket ε_i falls in

Arguments:
- N: number of observations (default: 100,000)
- J: number of choice alternatives (default: 4)

Returns: (Y, X) where Y is N×1 choice vector, X is N×K covariate matrix
"""
function sim_logit(N=100_000, J=4)
    # Generate X matrix (let's choose K=3 covariates)
    # Include intercept, and K-1 other covariates with different distributions
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    
    # Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    if J == 4
        β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    else
        # Generate random coefficients
        β = hcat(2 .* rand(size(X,2), J-1) .- 1, zeros(size(X,2)))
    end
    
    # Compute choice probabilities P (dimension N×J)
    # P_ij = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(X * β[:, j])./sum(exp.(X * β); dims=2)
    end
    
    # Draw uniform random variables
    draw = rand(N)
    
    # Generate choices based on cumulative probabilities
    # For each person i, find j such that:
    # Σ_{k=j}^J P_ik > ε_i
    Y = zeros(Int, N)
    for j = 1:J
        # Hint: sum(P[:,j:J]; dims=2) gives cumulative probabilities
        Y += vec(draw .< sum(P[:,j:J]; dims=2))
    end
    
    return Y, X
end

"""
    sim_logit_w_gumbel(N, J)

Simulate multinomial logit data using Gumbel shocks (alternative method).

This method directly uses the fact that if ε ~ Gumbel(0,1), then
    y_i = argmax_j (X_i'β_j + ε_ij)
generates choices from a multinomial logit model.

This is often simpler and more numerically stable than the inverse CDF method.
"""
function sim_logit_w_gumbel(N=100_000, J=4)
    # Generate X matrix (let's choose K=3 covariates)
    # Include intercept, and K-1 other covariates with different distributions
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    
    # Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    if J == 4
        β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    else
        # Generate random coefficients
        β = hcat(2 .* rand(size(X,2), J-1) .- 1, zeros(size(X,2)))
    end
    
    # Draw Gumbel errors
    ε = rand(Gumbel(0,1), N, J)
    
    # Choose alternative that maximizes utility
    # Y_i = argmax_j (X_i'β_j + ε_ij)
    # Hint: Use argmax.(eachrow(...))
    Y = argmax.(eachrow(X * β .+ ε))
    # X*β should be N x J
    # ε is also N x J
    
    return Y, X
end

################################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D)

Simulated Method of Moments for multinomial logit.

Algorithm:
1. For given parameters α, compute model-implied probabilities
2. Simulate D datasets using the Gumbel method
3. Compute average simulated choice frequencies
4. Match simulated frequencies to actual frequencies

Arguments:
- α: parameter vector
- X: covariate matrix
- y: actual choices
- D: number of simulation draws

Returns: SMM objective function value
"""
function mlogit_smm_overid(α, X, y, D)
    K = size(X, 2)
    J = length(α) ÷ K + 1  # α has K*(J-1) elements; counts alternatives even if some are never chosen
    N = length(y)
    
    @assert D >= 1 "D must be at least 1"

    # Create actual choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = (y .== j)
    end

    # Initialize simulated choice frequencies
    bigỸ = zeros(N, J)

    # Reshape parameters
    bigα = hcat(reshape(α, K, J-1), zeros(K))

    # Simulate D datasets and accumulate frequencies
    # Use identical random shocks at every optimizer evaluation.
    Random.seed!(1234)  # For reproducibility
    for d = 1:D
        # Draw Gumbel shocks, as in Question 3
        ε = rand(Gumbel(0,1), N, J)

        # Generate simulated choices
        ỹ = argmax.(eachrow(X * bigα .+ ε))

        # Update frequency counts
        for j = 1:J
            bigỸ[:,j] .+= (ỹ .== j) .* (1/D)
        end
    end

    # Compute moment vector (actual - simulated frequencies)
    g = vec(bigY .- bigỸ)

    # Compute objective function obj = N * g'g
    obj = N * g' * g

    return obj
end

################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################

"""
    main()

Main function that runs all estimation procedures.
Keeps everything out of global scope for better performance.
"""
function main()
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)
    
    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)
    
    # Estimate β using GMM
    # Hint: Use optimize() with LBFGS algorithm
    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage), 
                         rand(size(X_wage,2)), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    
    # Compare with closed-form OLS: (X'X)^(-1)X'y
    β_ols = X_wage \ y_wage
    
    println("GMM estimates: ", β_hat_gmm.minimizer)
    println("OLS estimates: ", β_ols)
    
    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)
    
    # Get starting values from series of binary logits
    # Create dummy variables for each occupation
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
    end
    
    # Run LPMs for each occupation
    svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        f = term(tempname) ~ term(:age) + term(:white) + term(:collgrad)
        svals[:,j] = coef(lm(f, df))
    end
    # Take differences relative to base category
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]
    
    # 2(a)
    # Estimate via MLE
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y), 
                         svals, 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # 2(b)
    # Estimate via GMM using MLE estimates as starting values
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                    α_hat_mle.minimizer, 
                                    LBFGS(), 
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("GMM estimates (MLE start): ", α_hat_gmm_mle_start.minimizer)
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # 2(c)
    # Estimate via GMM using random starting values
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                       rand(length(svals)), 
                                       LBFGS(), 
                                       Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("GMM estimates (random start): ", α_hat_gmm_random_start.minimizer)
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # Compare estimates and objective function values
    println("GMM objective (MLE start): ", α_hat_gmm_mle_start.minimum)
    println("GMM objective (random start): ", α_hat_gmm_random_start.minimum)
    println("Max abs difference in estimates: ",
            maximum(abs.(α_hat_gmm_mle_start.minimizer .- α_hat_gmm_random_start.minimizer)))
    
    #--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)
    
    # Simulate data
    ySim, XSim = sim_logit(100_000, 4)
    ySimG, XSimG = sim_logit_w_gumbel(100_000, 4)
    
    println("Note: X and ε drawn separately so they won't match exactly")
    println("but with same β and distribution of X and large N they should be close")
    println("freq of j==1 in ySim: ",mean(ySim.==1))
    println("freq of j==1 in ySimG: ",mean(ySimG.==1))
    
    # Estimate parameters from simulated data and compare method
    α_hat_sim = optimize(a -> mlogit_mle(a, XSim, ySim), 
                         rand(9),  # 3 covariates × 3 non-base alternatives
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    
    # Try alternative simulation method with Gumbel
    α_hat_simG = optimize(a -> mlogit_mle(a, XSimG, ySimG), 
                         rand(9),  # 3 covariates × 3 non-base alternatives
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    
    println("True β used in simulation:", hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3)))
    println("Estimated β from simulated data:")
    println("Method 1 (sim_logit): ", α_hat_sim.minimizer)
    println("Method 2 (sim_logit_w_gumbel): ", α_hat_simG.minimizer)
    
    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)
    
    # Estimate via SMM
    # Note: This will be slow! Start with small D (like 100) for testing
    # Then increase to 1000-2000 for final estimates
    D = parse(Int, get(ENV, "PS7_D_SMM", "100"))

    # For simulated hard-choice indicators, the objective is stepwise in α.
    # Unlike the LBFGS hint, NelderMead does not require unreliable gradients.
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, D),
                         α_hat_mle.minimizer,  # Use MLE as starting values
                         NelderMead(),
                         Optim.Options(iterations=1000, show_trace=true))

    # Compare SMM estimates to MLE and GMM estimates
    println("MLE estimates: ", α_hat_mle.minimizer)
    println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
    println("SMM estimates: ", α_hat_smm.minimizer)
    println("SMM objective: ", Optim.minimum(α_hat_smm))
    println("SMM converged: ", Optim.converged(α_hat_smm))
    # With finite D, the SMM criterion is noisy/stepwise, so SMM estimates
    # may differ from smooth MLE and GMM estimates. Increase D for a final run.

    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)
    
    # Return estimates for testing
    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm
end
