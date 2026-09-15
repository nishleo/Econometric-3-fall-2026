#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function q1()
    function f(x)
    return -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    end
 # f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
  # minimize -f(x) to find the maximum of f(x)
  minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
  startval = rand(1)   # random starting value
  result = optimize(minusf, startval, LBFGS())
  println("optimization result is ",result) #for extra data
  println("argmin (minimizer) is ",Optim.minimizer(result)[1])
  println("min is ",Optim.minimum(result))
  println("max is ",-Optim.minimum(result))

  return result

end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function q2()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y # check using matrix formula
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)


    println("OLS using formula:")
    println(bols)

    return beta_hat_ols.minimizer

end

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function q3()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1


    function logit(beta, X, y)

        xb = X*beta

        p = 1 ./ (1 .+ exp.(-xb))

        loglik = sum(
            y .* log.(p) +
            (1 .- y) .* log.(1 .- p)
        )

        return -loglik
    end


    beta_hat_logit = optimize(
        b -> logit(b, X, y),
        zeros(size(X,2)),
        LBFGS()
    )

    println("Logit using Optim:")
    println(beta_hat_logit.minimizer)

    return beta_hat_logit.minimizer
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function q4()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

    df = CSV.read(HTTP.get(url).body, DataFrame)

    df.white = df.race .== 1

    model = glm(
        @formula(married ~ age + white + collgrad),
        df,
        Binomial(),
        LogitLink()
    )

    println("Logit using GLM:")
    println(coef(model))

    return coef(model)
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function q5()

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    df = dropmissing(df, :occupation) # remove missing occupation
    
    freqtable(df, :occupation) # note small number of obs in some occupations

    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved


    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = Int.(df.occupation)
    K = size(X,2)


    function mlogit(beta, X, y)

        # reshape 24 parameters into 4 x 6
        B = reshape(beta, K, 6)

        loglik = 0.0
        for i in 1:length(y)
            V = X[i,:]' * B # utilities for alternatives 1-6
            V = vec(V)
            # occupation 7 is base, utility = 0
            V = [V; 0.0]
            denominator = sum(exp.(V))
            probability = exp(V[y[i]]) / denominator
            loglik += log(probability)
        end

        return -loglik
    end

    startval = zeros(K*6)


    beta_hat_mlogit = optimize( b -> mlogit(b, X, y), startval, LBFGS(), Optim.Options( g_tol=1e-5, iterations=100_000        ))

    println("Multinomial logit estimates:")
    println( reshape(beta_hat_mlogit.minimizer, K, 6) )

    return beta_hat_mlogit.minimizer
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function runall()

    println("\nQUESTION 1")
    q1()

    println("\nQUESTION 2")
    q2()

    println("\nQUESTION 3")
    q3()

    println("\nQUESTION 4")
    q4()

    println("\nQUESTION 5")
    q5()

end

