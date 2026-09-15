using Test
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_NISHANT_SOURCE.jl")


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# test question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@testset "Question 1" begin

    result = q1()

    xhat = Optim.minimizer(result)[1]
    maxf = -Optim.minimum(result)

    @test isapprox(xhat, -7.37824, atol=0.001)
    @test isapprox(maxf, 964.313, atol=0.01)

end


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# test question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@testset "Question 2" begin

    beta_optim = q2()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_formula = inv(X'*X)*X'*y

    @test length(beta_optim) == 4
    @test isapprox(beta_optim, beta_formula, atol=1e-4)

end


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# test questions 3 and 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@testset "Questions 3 and 4" begin

    beta_optim = q3()
    beta_glm = q4()

    @test length(beta_optim) == 4
    @test length(beta_glm) == 4

    # Optim logit should give same estimates as GLM
    @test isapprox(beta_optim, beta_glm, atol=1e-4)

end


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# test question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@testset "Question 5" begin

    beta = q5()

    # 4 regressors x 6 estimated alternatives
    @test length(beta) == 24

    # all estimated coefficients should be finite
    @test all(isfinite.(beta))


    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    df = dropmissing(df, :occupation)

    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = Int.(df.occupation)

    K = size(X,2)


    function mlogit_test(beta, X, y)

        B = reshape(beta, K, 6)

        loglik = 0.0

        for i in 1:length(y)

            V = X[i,:]' * B
            V = vec(V)
            V = [V; 0.0]

            denominator = sum(exp.(V))
            probability = exp(V[y[i]]) / denominator

            loglik += log(probability)

        end

        return -loglik
    end


    # estimated model should fit better than all coefficients = 0
    nll_zero = mlogit_test(zeros(K*6), X, y)
    nll_estimated = mlogit_test(beta, X, y)

    @test nll_estimated < nll_zero

end