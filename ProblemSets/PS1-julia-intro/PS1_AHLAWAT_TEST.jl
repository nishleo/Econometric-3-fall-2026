using Test

include("PS1_AHLAWAT_SOURCE.jl")


# =============================================================
# Tests for Question 1
# =============================================================

@testset "Question 1 tests" begin

    A, B, C, D = q1()


    # Test dimensions
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)


    # A should be between -5 and 10
    @test all(A .>= -5)
    @test all(A .<= 10)


    # C should contain the specified parts
    # of A and B

    @test C[:, 1:5] == A[1:5, 1:5]
    @test C[:, 6:7] == B[1:5, 6:7]


    # D should equal A whenever A <= 0
    # and zero otherwise

    @test D == ifelse.(A .<= 0, A, 0.0)


    # A should contain 70 elements

    @test length(A) == 70


    # Check that output files were created

    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")

end


# =============================================================
# Tests for Question 2
# =============================================================

@testset "Question 2 tests" begin

    A, B, C, D = q1()

    # q2 is explicitly supposed to return nothing.
    result = q2(A, B, C)

    @test result === nothing

end