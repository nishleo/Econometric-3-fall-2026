# you should have a unit test to verify that transition matrix is actually markov

using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include("create_grids.jl")
include("PS5_NISHANT_SOURCE.jl")

@testset "Transition matrix is Markov" begin
    zval, zbin, xval, xbin, xtran = create_grids()
    @test zbin == 101
    @test xbin == 201
    @test size(xtran) == (zbin*xbin, xbin)
    @test xval == collect(0.0:0.125:25.0)
    @test zval == collect(0.25:0.01:1.25)
    @test all(isfinite, xtran)
    @test all(p -> 0.0 <= p <= 1.0, xtran)
    # Every row must sum to one, not just the average row.
    for row in 1:size(xtran,1)
        @test sum(xtran[row,:]) ≈ 1.0 atol=1e-12
    end
    # Driving cannot reduce mileage. The final bin includes the entire tail.
    for z in 1:zbin, x in 1:xbin
        row = x + (z-1)*xbin
        @test all(iszero, xtran[row,1:x-1])
        @test xtran[row,end] ≈ exp(-zval[z]*(25.0-xval[x])) atol=1e-12
    end
    # Check the bin probabilities against equation (3).
    for z in (1,51,101), x in (1,100,200), k in x:200
        expected = exp(-zval[z]*(xval[k]-xval[x])) -
                   exp(-zval[z]*(xval[k]+0.125-xval[x]))
        @test xtran[x+(z-1)*xbin,k] ≈ expected atol=1e-12
    end
    # Each route block is a square Markov matrix; two-step rows also sum to one.
    for z in (1,51,101)
        P = xtran[(z-1)*xbin+1:z*xbin,:]
        @test P[end,end] == 1.0
        @test vec(sum(P*P,dims=2)) ≈ ones(xbin) atol=1e-12
    end
end

@testset "Backward recursion and dynamic likelihood" begin
    # Small panel with both brands and two different route transition matrices.
    d = (Y=[1 0 1; 0 1 0; 1 1 0; 0 0 1],
         X=[0.0 0.5 1.0; 1.0 0.5 0.0; 0.5 1.0 0.0; 1.0 0.0 0.5],
         Xstate=[1 2 3; 3 2 1; 2 3 1; 3 1 2], Zstate=[1,1,2,2],
         B=[0,1,0,1], N=4, T=3, xval=[0.0,0.5,1.0], xbin=3, zbin=2,
         xtran=[0.6 0.3 0.1; 0.0 0.7 0.3; 0.0 0.0 1.0;
                0.8 0.15 0.05; 0.0 0.9 0.1; 0.0 0.0 1.0], β=0.9)
    θ = [1.0,-0.7,0.4]
    for β in (0.0,0.5,0.9,1.0)
        data = merge(d, (β=β,))
        FV = fill(99.0,6,2,4) # Also tests resetting a reused terminal slice.
        @test compute_future_value!(FV,θ,data) === nothing
        @test FV[:,:,end] == zeros(6,2)
        @test all(isfinite,FV)

        # Independent reference: V is UNDISCOUNTED, whereas FV stores β*V.
        V = zeros(6,2,4)
        for t in 3:-1:1, b in 0:1, z in 1:2, x in 1:3
            row0 = 1+(z-1)*3
            row = x+(z-1)*3
            v0 = β*sum(data.xtran[row0,k]*V[row0+k-1,b+1,t+1] for k in 1:3)
            v1 = θ[1]+θ[2]*data.xval[x]+θ[3]*b +
                 β*sum(data.xtran[row,k]*V[row0+k-1,b+1,t+1] for k in 1:3)
            V[row,b+1,t] = log(exp(v0)+exp(v1))
        end
        @test FV ≈ β.*V atol=1e-11
        reference_nll = 0.0
        for i in 1:4, t in 1:3
            row0 = 1+(data.Zstate[i]-1)*3
            row1 = data.Xstate[i,t]+(data.Zstate[i]-1)*3
            value = θ[1]+θ[2]*data.X[i,t]+θ[3]*data.B[i] + β*sum(
                (data.xtran[row1,k]-data.xtran[row0,k])*V[row0+k-1,data.B[i]+1,t+1] for k in 1:3)
            p = exp(value)/(1+exp(value))
            reference_nll -= data.Y[i,t] == 1 ? log(p) : log(1-p)
        end
        @test log_likelihood_dynamic(θ,data) ≈ reference_nll atol=1e-11
    end
    @test log_likelihood_dynamic(zeros(3),d) ≈ d.N*d.T*log(2)
    # β=0 must reduce to the ordinary static logit likelihood.
    static_nll = sum(log(1+exp(θ[1]+θ[2]*d.X[i,t]+θ[3]*d.B[i])) -
        d.Y[i,t]*(θ[1]+θ[2]*d.X[i,t]+θ[3]*d.B[i]) for i in 1:4,t in 1:3)
    @test log_likelihood_dynamic(θ,merge(d,(β=0.0,))) ≈ static_nll
    # No mileage effect means continuation differences cancel.
    @test log_likelihood_dynamic([1.0,0.0,0.4],d) ≈
          log_likelihood_dynamic([1.0,0.0,0.4],merge(d,(β=0.0,)))
    for extreme in ([1000.0,-1000.0,1000.0],[-1000.0,1000.0,-1000.0])
        @test isfinite(log_likelihood_dynamic(extreme,d))
    end
    original = deepcopy(d)
    @test log_likelihood_dynamic(θ,d) == log_likelihood_dynamic(θ,d)
    @test d == original
end

@testset "Static estimation and optimization wrapper" begin
    # Exact cell odds imply θ = [log(2), -log(2), log(2)].
    Y, X, B = Int[], Float64[], Int[]
    for (x,b,yes,no) in ((0.0,0,100,50),(1.0,0,100,100),(0.0,1,100,25),(1.0,1,100,50))
        append!(Y,vcat(ones(Int,yes),zeros(Int,no)))
        append!(X,fill(x,yes+no))
        append!(B,fill(b,yes+no))
    end
    df_long = DataFrame(Y=Y,Odometer=X,Branded=B)
    fit = estimate_static_model(df_long)
    θ_true = [log(2),-log(2),log(2)]
    @test coef(fit) ≈ θ_true atol=1e-5
    N = length(Y)
    d = (Y=reshape(Y,N,1), X=reshape(X,N,1), B=B,
         Xstate=reshape(Int.(X).+1,N,1), Zstate=ones(Int,N), N=N,T=1,
         xval=[0.0,1.0],xbin=2,zbin=1,xtran=[0.6 0.4;0.0 1.0],β=0.0)
    @test -log_likelihood_dynamic(coef(fit),d) ≈ loglikelihood(fit) atol=1e-9
    result = estimate_dynamic_model(d;θ_start=zeros(3))
    @test result !== nothing
    @test Optim.converged(result)
    @test Optim.minimizer(result) ≈ θ_true atol=1e-5
    @test Optim.minimum(result) <= log_likelihood_dynamic(zeros(3),d)
end

# Optional integration tests use the original loading functions unchanged.
if "--integration" in ARGS
    @testset "Assignment data loading" begin
        df_long = load_static_data()
        @test nrow(df_long) == 20000
        @test length(unique(zip(df_long.bus_id,df_long.time))) == 20000
        @test issorted(collect(zip(df_long.bus_id,df_long.time)))
        @test all(y -> y in (0,1), df_long.Y)
        @test all(isfinite,df_long.Odometer)
        @test all(isfinite,coef(estimate_static_model(df_long)))
        d = load_dynamic_data()
        @test (d.N,d.T) == (1000,20)
        @test size(d.Y) == size(d.X) == size(d.Xstate) == (1000,20)
        @test length(d.B) == length(d.Zstate) == 1000
        @test all(x -> x isa Integer && 1 <= x <= d.xbin,d.Xstate)
        @test all(z -> z isa Integer && 1 <= z <= d.zbin,d.Zstate)
        @test all(b -> b in (0,1),d.B)
        @test all(y -> y in (0,1),d.Y)
        @test isfinite(log_likelihood_dynamic([2.0,-0.15,1.0],d))
    end
end
