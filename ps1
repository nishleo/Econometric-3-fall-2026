using Random
using LinearAlgebra
using JLD
using DataFrames
using CSV


function q1()

    # ---------------------------------------------------------
    # (a) Create matrices A, B, C, and D
    # ---------------------------------------------------------

    Random.seed!(1234)


    # (i) A: 10 x 7, Uniform[-5, 10]
    # rand() produces Uniform[0,1], so transform:
    # -5 + 15*rand()

    A = -5 .+ 15 .* rand(10, 7)


    # (ii) B: 10 x 7, Normal(mean = -2, std = 15)
    # randn() produces N(0,1)

    B = -2 .+ 15 .* randn(10, 7)


    # (iii) C: 5 x 7
    # First 5 rows & first 5 columns from A
    # Last 2 columns & first 5 rows from B

    C = hcat(A[1:5, 1:5], B[1:5, 6:7])


    # (iv) D: 10 x 7
    # D[i,j] = A[i,j] if A[i,j] <= 0
    # otherwise D[i,j] = 0

    D = ifelse.(A .<= 0, A, 0.0)


    # ---------------------------------------------------------
    # (b) Number of elements of A
    # ---------------------------------------------------------

    println("Number of elements in A: ", length(A))


    # ---------------------------------------------------------
    # (c) Number of unique elements of D
    # ---------------------------------------------------------

    println("Number of unique elements in D: ", length(unique(D)))


    # ---------------------------------------------------------
    # (d) vec(B) using reshape
    # ---------------------------------------------------------

    E = reshape(B, :, 1)

    # Easier way:
    # E = vec(B)
    #
    # Note:
    # reshape(B, :, 1) gives a 70 x 1 matrix
    # vec(B) gives a length-70 vector


    # ---------------------------------------------------------
    # (e) 3-dimensional array F
    # F[:,:,1] = A
    # F[:,:,2] = B
    # ---------------------------------------------------------

    F = cat(A, B, dims=3)

    println("Size of F before permutation: ", size(F))


    # ---------------------------------------------------------
    # (f) Change F from 10 x 7 x 2 to 2 x 10 x 7
    # ---------------------------------------------------------

    F = permutedims(F, (3, 1, 2))

    println("Size of F after permutation: ", size(F))


    # ---------------------------------------------------------
    # (g) Kronecker product
    # ---------------------------------------------------------

    G = kron(B, C)

    println("Size of G: ", size(G))

    # C ⊗ F:
    #
    # kron(C, F)
    #
    # This will cause a problem because F is a 3-dimensional
    # array, while the usual matrix Kronecker product is
    # defined here for vectors/matrices.


    # ---------------------------------------------------------
    # (h) Save A, B, C, D, E, F, G
    # ---------------------------------------------------------

    save(
        "matrixpractice.jld",
        "A", A,
        "B", B,
        "C", C,
        "D", D,
        "E", E,
        "F", F,
        "G", G
    )


    # ---------------------------------------------------------
    # (i) Save only A, B, C, D
    # ---------------------------------------------------------

    save(
        "firstmatrix.jld",
        "A", A,
        "B", B,
        "C", C,
        "D", D
    )


    # ---------------------------------------------------------
    # (j) Export C as Cmatrix.csv
    # ---------------------------------------------------------

    Cdf = DataFrame(C, :auto)

    CSV.write("Cmatrix.csv", Cdf)


    # ---------------------------------------------------------
    # (k) Export D as tab-delimited Dmatrix.dat
    # ---------------------------------------------------------

    Ddf = DataFrame(D, :auto)

    CSV.write("Dmatrix.dat", Ddf, delim='\t')


    # ---------------------------------------------------------
    # (l) Return A, B, C, D
    # ---------------------------------------------------------

    return A, B, C, D

end


# Run q1() and save its four outputs
A, B, C, D = q1()