using Random, LinearAlgebra, Statistics, JLD, DataFrames, CSV, Distributions

function q1()
    Random.seed!(1234)

    A = -5 .+ 15 .* rand(10, 7)
    B = -2 .+ 15 .* randn(10, 7)
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    D = ifelse.(A .<= 0, A, 0.0)

    println("Number of elements in A: ", length(A))
    println("Number of unique elements in D: ", length(unique(D)))

    E = reshape(B, :, 1)
    F = cat(A, B, dims=3)
    println("Size of F before permutation: ", size(F))

    F = permutedims(F, (3, 1, 2))
    println("Size of F after permutation: ", size(F))

    G = kron(B, C)
    println("Size of G: ", size(G))

    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    CSV.write("Cmatrix.csv", DataFrame(C, :auto))
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    return A, B, C, D
end


function q2(A, B, C)
    AB = zeros(size(A))

    for r in axes(A, 1), c in axes(A, 2)
        AB[r, c] = A[r, c] * B[r, c]
    end

    AB2 = A .* B
    println("AB and AB2 are equal: ", isequal(AB, AB2))

    Cprime = Float64[]

    for c in axes(C, 2), r in axes(C, 1)
        if -5 <= C[r, c] <= 5
            push!(Cprime, C[r, c])
        end
    end

    Cprime2 = C[(C .>= -5) .& (C .<= 5)]
    println("Cprime and Cprime2 are equal: ", isequal(Cprime, Cprime2))

    N, K, T = 15_169, 6, 5
    X = zeros(N, K, T)

    for i in 1:N
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))

        for t in 1:T
            X[i, 2, t] = rand() <= 0.75 * (6 - t) / 5 ? 1.0 : 0.0

            mean3, sd3 = 15 + t - 1, 5 * (t - 1)
            X[i, 3, t] = sd3 == 0 ? mean3 : rand(Normal(mean3, sd3))

            mean4, sd4 = π * (6 - t) / 3, 1 / exp(1)
            X[i, 4, t] = rand(Normal(mean4, sd4))
        end
    end

    println("Size of X: ", size(X))

    β = zeros(K, T)
    β[1, :] = [1 + 0.25 * (t - 1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t / 3 for t in 1:T]

    println("Size of beta: ", size(β))
    display(β)

    Y = zeros(N, T)

    for t in 1:T
        ε = rand(Normal(0, 0.36), N)
        Y[:, t] = X[:, :, t] * β[:, t] + ε
    end

    println("Size of Y: ", size(Y))

    return nothing
end