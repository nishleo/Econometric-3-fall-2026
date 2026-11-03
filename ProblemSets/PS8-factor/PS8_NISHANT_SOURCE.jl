#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return as a DataFrame.

# Arguments
- `url::String`: URL to the CSV file

# Returns
- DataFrame containing the NLSY data
"""
function load_data(url::String)
    return CSV.read(HTTP.get(url).body, DataFrame)
end

"""
    estimate_base_regression(df::DataFrame)

Estimate the baseline wage regression without ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()
"""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Compute the correlation matrix for the six ASVAB test scores.

# Arguments
- `df::DataFrame`: Data containing the six ASVAB scores

# Returns
- DataFrame with correlation matrix (6x6)

# Hint
- The ASVAB scores are in columns: asvabAR, asvabCS, asvabMK, asvabNO, asvabPC, asvabWK
- Select these columns by name (e.g. df[:, [:asvabAR, :asvabCS, ...]]) rather than position
- Use cor() function on the matrix of ASVAB scores
"""
function compute_asvab_correlations(df::DataFrame)
    asvabMat = Matrix(df[!, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])
    corrMat = cor(asvabMat)
    return DataFrame(corrMat, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]; copycols=false)
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Estimate wage regression including all six ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()

# Question to consider:
- Given the correlations you computed, what problems might arise?
"""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr +
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    estimate_pca_regression(df::DataFrame)

Perform PCA on ASVAB scores and include first principal component in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first principal component

# Key steps:
1. Extract ASVAB scores as a matrix
2. IMPORTANT: Transpose to J×N (MultivariateStats requires features × observations)
3. Fit PCA model with maxoutdim=1
4. Transform the data to get principal component scores
5. Add PC scores to DataFrame and run regression

# Hints:
- asvabMat = Matrix(df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])'
  (note the transpose!)
- M = fit(PCA, asvabMat; maxoutdim=1)
- asvabPCA = MultivariateStats.transform(M, asvabMat)
- asvabPCA will be 1×N, need to reshape for regression
- The sign of a principal component is arbitrary, so interpret its coefficient with care
"""
function estimate_pca_regression(df::DataFrame)
    asvabMat = transpose(Matrix(df[!, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]))
    
    M = fit(PCA, asvabMat; maxoutdim=1)
    
    asvabPCA = MultivariateStats.transform(M, asvabMat)
    
    df[!, :PC] = vec(asvabPCA)

    
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + PC), df)
    
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df::DataFrame)

Perform Factor Analysis on ASVAB scores and include first factor in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first factor

# Note:
- Syntax is nearly identical to PCA, just use FactorAnalysis instead
- Make sure you analyze only the six ASVAB scores. If you added the PCA
  column to df in Question 4, "the last 6 columns" now includes it!
"""
function estimate_factor_regression(df::DataFrame)
    asvabMat = transpose(Matrix(df[!, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]))
    M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    asvabFA = MultivariateStats.transform(M, asvabMat)
    df[!, :FA] = vec(asvabFA)

    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + FA), df)
end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

"""
    prepare_factor_matrices(df::DataFrame)

Prepare the data matrices needed for the factor model estimation.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- `X`: Covariates for wage equation (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: Log wage outcomes (N×1)
- `Xfac`: Covariates for measurement equations (N×4: black, hispanic, female, constant)
- `asvabs`: Matrix of all 6 ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    X = Matrix{Float64}(undef, nrow(df), 7)
    for (j, name) in enumerate((:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr))
        copyto!(view(X, :, j), df[!, name])
    end
    fill!(view(X, :, 7), 1.0)
    
    y = df[!, :logwage]
    
    Xfac = Matrix{Float64}(undef, nrow(df), 4)
    copyto!(view(Xfac, :, 1:3), view(X, :, 1:3))
    fill!(view(Xfac, :, 4), 1.0)
    
    asvabs = Matrix{Float64}(undef, nrow(df), 6)
    for (j, name) in enumerate((:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK))
        copyto!(view(asvabs, :, j), df[!, name])
    end
    
    return X, y, Xfac, asvabs
end

"""
    factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                 y::Vector, R::Integer) where T<:Real

Compute the negative log-likelihood for the factor model.

# Arguments
- `θ`: Parameter vector containing:
  * α parameters (L×J matrix, vectorized): coefficients in measurement equations
  * β parameters (K×1 vector): coefficients in wage equation  
  * γ parameters (J vector): factor loadings in measurement equations
  * δ parameter (scalar): factor loading in wage equation
  * σ parameters (J+1 vector): standard deviations (J for measurements, 1 for wage)
- `X`: Wage equation covariates (N×K)
- `Xfac`: Measurement equation covariates (N×L)
- `Meas`: ASVAB test scores (N×J)
- `y`: Log wages (N×1)
- `R`: Number of quadrature points

# Returns
- Negative log-likelihood value (scalar)

# Model Structure:
## Measurement equations (for each j=1,...,J):
   asvab_j = Xfac*α_j + γ_j*ξ + ε_j,  ε_j ~ N(0, σ_j²)

## Wage equation:
   logwage = X*β + δ*ξ + ε,  ε ~ N(0, σ_{J+1}²)

## Latent factor:
   ξ ~ N(0,1)

# Likelihood for person i:
   L_i = ∫ [∏_j φ((M_ij - Xfac_i*α_j - γ_j*ξ)/σ_j) / σ_j] 
          × [φ((y_i - X_i*β - δ*ξ)/σ_{J+1}) / σ_{J+1}]
          × φ(ξ) dξ

# Key Steps:
1. Unpack θ into α, β, γ, δ, σ parameters
2. Set up Gauss-Legendre quadrature nodes and weights
3. For each quadrature point:
   a. Compute likelihood contribution from each measurement equation
   b. Compute likelihood contribution from wage equation
   c. Weight by quadrature weight and ξ density
4. Sum log-likelihoods across all observations
5. Return negative for minimization
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                     y::Vector, R::Integer) where T<:Real
    
    # Get dimensions
    K = size(X, 2)      # Number of covariates in wage equation
    L = size(Xfac, 2)   # Number of covariates in measurement equations
    J = size(Meas, 2)   # Number of ASVAB tests
    N = length(y)       # Number of observations
    
    α_end = L * J
    β_end = α_end + K
    γ_end = β_end + J
    α = reshape(view(θ, 1:α_end), L, J)
    β = view(θ, (α_end + 1):β_end)
    γ = view(θ, (β_end + 1):γ_end)
    δ = θ[γ_end + 1]
    σ = view(θ, (γ_end + 2):(γ_end + J + 2))
    
    ξ, ω = lgwt(R, -5, 5)
    
    # Initialize likelihood storage
    like = zeros(T, N)
    
    measurement_residuals = Meas - Xfac * α
    wage_residuals = y - X * β
    invσ = inv.(σ)
    scaled_γ = γ .* invσ[1:J]
    scaled_δ = δ * invσ[J + 1]
    normalizer = (inv(sqrt(2π)))^(J + 2) * prod(invσ)
    for j in 1:J
        for i in 1:N
            measurement_residuals[i, j] *= invσ[j]
        end
    end
    wage_residuals .*= invσ[J + 1]
    squared_residuals = similar(like)

    for r in 1:R
        node = ξ[r]
        wage_shift = scaled_δ * node
        for i in 1:N
            squared_residuals[i] = abs2(wage_residuals[i] - wage_shift)
        end
        for j in 1:J
            measurement_shift = scaled_γ[j] * node
            for i in 1:N
                squared_residuals[i] += abs2(measurement_residuals[i, j] - measurement_shift)
            end
        end
        weight = ω[r] * normalizer * exp(-abs2(node) / 2)
        for i in 1:N
            like[i] += weight * exp(-squared_residuals[i] / 2)
        end
    end
    
    return -sum(log, like)
end

"""
    run_estimation(df::DataFrame, start_vals::Vector)

Run the full MLE estimation procedure for the factor model.

# Arguments
- `df::DataFrame`: The data
- `start_vals::Vector`: Starting values for optimization

# Returns
- `θ̂`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood at optimum

# Steps:
1. Prepare data matrices
2. Set up TwiceDifferentiable objective (for Hessian-based optimization)
3. Optimize using Newton method with line search
4. Compute standard errors from inverse Hessian
"""
function run_estimation(df::DataFrame, start_vals::Vector)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    
    objective = TwiceDifferentiable(
        θ -> factor_model(θ, X, Xfac, asvabs, y, 50),
        start_vals;
        autodiff = Optim.ADTypes.AutoForwardDiff(),
    )
    
    result = Optim.optimize(
        objective,
        start_vals,
        Optim.Newton(linesearch = BackTracking()),
        Optim.Options(g_tol = 1e-8, iterations = 1_000, show_trace = false),
    )
    
    θ̂ = Optim.minimizer(result)
    H = Symmetric(Optim.hessian!(objective, θ̂))
    se = sqrt.(diag(inv(cholesky(H))))
    
    return θ̂, se, -Optim.minimum(result)
    
end

"""
    format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)

Format estimation results into a readable DataFrame.

# Arguments
- `θ`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood value
- `asvabs`: ASVAB matrix (used to get J)

# Returns
- DataFrame with columns: equation, variable, coefficient, std_error
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    J = size(asvabs, 2)
    asvab_names = ("asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK")
    measurement_names = [j <= length(asvab_names) ? asvab_names[j] : "asvab$j" for j in 1:J]
    equation_names = vcat(
        repeat(measurement_names; inner = 4),
        fill("logwage", 7),
        measurement_names,
        ["logwage"],
        measurement_names,
        ["logwage"],
    )
    variable_names = vcat(
        repeat(["black", "hispanic", "female", "constant"], J),
        ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", "constant"],
        fill("factor_loading", J + 1),
        fill("sigma", J + 1),
    )
    
    return DataFrame(
        equation = equation_names,
        variable = variable_names,
        coefficient = θ,
        std_error = se,
    )
    
end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main()

Execute the complete analysis workflow for Problem Set 8.

This function runs through all questions sequentially:
1. Load data and run base regression
2. Compute ASVAB correlations
3. Run full regression with all ASVABs
4. Run PCA regression
5. Run Factor Analysis regression
6. Estimate full factor model via MLE
"""
function main()
    # Data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    
    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)
    
    # Load data
    println("\nLoading data...")
    df = load_data(url)
    println("Data loaded successfully. Dimensions: ", size(df))
    
    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    display(estimate_base_regression(df))
    
    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    display(compute_asvab_correlations(df))
    println("Consider: Are these correlations high? What might this imply?")
    
    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    display(estimate_full_regression(df))
    println("Consider: How do results compare to Question 1? Any concerns?")
    
    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    display(estimate_pca_regression(df))
    
    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    display(estimate_factor_regression(df))
    
    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    J = size(asvabs, 2)
    α_start = Xfac \ asvabs
    β_start = X \ y
    γ_start = fill(0.5, J)
    δ_start = 0.5
    σ_start = fill(0.5, J + 1)
    start_vals = vcat(vec(α_start), β_start, γ_start, δ_start, σ_start)
    
    println("\nEstimating full factor model...")
    θ̂, se, loglike = run_estimation(df, start_vals)
    
    display(format_results(θ̂, se, loglike, asvabs))
    println("Log-likelihood at optimum: ", loglike)
    
    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
end