using Test, NamedPlus
import NamedDims

v = NamedDimsArray(rand(3), :j)
m = NamedDimsArray(rand(2,3), (:i,:j))
t = NamedDimsArray(rand(2,3,4), (:i,:j,:k))
z = NamedDimsArray(rand(26), :z)

@testset "macro & names" begin

    @named begin
        v2{j} = rand(3)
        m2{i,j} = rand(2,3)
        t2{i,j,k} = rand(2,3,4)
    end
    @test NamedDims.names(v2) == (:j,)
    @test_skip names(t2,2) == :j

    @test (@named [i^2 for i in 1:3]) isa NamedDimsArray
    @test NamedDims.names(@named [i/j for i in 1:3, j in 1:4]) == (:i, :j)

    @test_skip ranges(@named [i^2 for i in 1:2:5]) == (1:2:5,)

end

@testset "unname with names" begin

    @test unname(m, (:j, :i)) === transpose(parent(m))

    @test unname(t, (:i,:j,:k)) === parent(t)
    @test unname(t, (:i,:k,:j)) === PermutedDimsArray(parent(t), (1,3,2))

    @test NamedDims.names(permutenames(m, (:i, :k, :j))) == (:i, :_, :j)
    @test NamedDims.names(permutenames(v, (:i, :j, :k))) == (:_, :j, :_)

end

# broken without TransmuteDims master?
@testset "broadcasting by name" begin

    @test NamedDims.names(m ./ v') == (:i, :j)
    @test_throws DimensionMismatch m ./ v

    @test NamedDims.names(@named w{i,k,j} = t .+ m ./ v) == (:i, :k, :j)
    @named z{i,j,k} = t .+ m ./ v
    @test z == t .+ m ./ v'

end

@testset "rename & prime" begin

    @test prime(:z) == :z′
    @test NamedDims.names(prime(m, first)) == (:i′, :j)
    @test NamedDims.names(prime(m, 2)) == (:i, :j′)

    @test NamedDims.names(rename(m, :j => :k)) == (:i, :k)
    @test NamedDims.names(rename(m, (:a, :b))) == (:a, :b)

    using NamedPlus: _prime

    @test (@inferred (() -> prime(:a))() ;true)
    @test (@inferred (() -> _prime((:i,:j,:k), Val(1)))() ;true)

end
@testset "split & join" begin

    @test names(join(t, (:i,:j) => :ij)) == (:ij, :k)
    t1 = join(t, :i,:k)
    @test names(t1) == (:j, Symbol("i⊗k"))

    @test size(split(m, :i)) == (2, 1, 3)
    @test size(split(m, :i => (:i1, :i2))) == (2, 1, 3)
    @test size(split(m, :i => (:i1, :i2), (1,2))) == (1, 2, 3)
    @test size(split(m, :i)) == (2, 1, 3)

    @test t == split(join(t, (:i,:j) => :ij), :ij => (:i,:j), (2,3))

    t2 = split(t1, (:i,:k), (2,4));
    @test names(t2) == (:j, :i, :k)
    t2 = split(t1, (:i,:k), (2,4))
    @test size(t2) == (3, 2, 4)

    t2[1,1,1] = 99
    @test t1[1,1,1] == 99
    @test t[1,1,1] != 99

    using NamedPlus: _join, _split

    @test (@inferred (() -> _join(:i, :j))() ;true)
    @test (@inferred (() -> _split(_join(:i, :j)))() ;true)

end

using LinearAlgebra, OMEinsum, TensorOperations

@testset "wrapper types" begin

    d = Diagonal(v)
    @test getnames(d) == (:j, :j)
    @test typeof(nameless(d)) == Diagonal{Float64,Array{Float64,1}}
    @test d[j=2] === v[2] # indexing of NamedUnion

    @test canonise(d) === v

    d2 = diagonal(v, (:j, :j′))
    d3 = diagonal(v, (:j, :j))
    @test d2 isa NamedDimsArray
    @test d3[j=2] === v[2]
    @test canonise(d2) isa NamedDimsArray{(:j, :j′),Float64,2,<:Diagonal}
    @test canonise(d3) === v

    p = PermutedDimsArray(t, (3,1,2));
    @test p == PermutedDimsArray(t, (:k,:i,:j))
    @test getnames(p) == (:k, :i, :j)
    # @test summary(p) == "k≤4 × i≤2 × j≤3 PermutedDimsArray{Float64,3,(3, 1, 2),(2, 3, 1),NamedDimsArray{(:i, :j, :k),Float64,3,Array{Float64,3}}}"
    @test t === canonise(p)

    @test p[i=1, k=3, j=2] == t[1,2,3]
    @test p[i=1, k=3] == t[1,:,3]
    @test p[i=1] == transpose(t[1,:,:]) # is this OK?

end
@testset "SVD" begin

    for f in (identity, transpose, permutedims)
        s = svd(f(m))
        c = contract(s.U, s.S, s.V; dims=:svd)
        @test names(c) == names(f(m))
        @test f(m) ≈ c
    end

end
#= broken
@testset "generalised contraction" begin

    *ⱼ(x...) = NamedPlus.Contract{(:j,)}(x...)

    @test names(t *ⱼ m) == (:i,:k)
    @test names(t *ⱼ diagonal(v)) == (:i,:k)
    @test names(t *ⱼ diagonal(v, (:j, :j′))) == (:i,:k,:j′)

end
=#
@testset "tensor macro" begin

    for f in (identity, transpose, permutedims)
        tm = f(m)

        @named @tensor w[k] := t[i,j,k] * tm[i,j]
        @test names(w) == (:k,)

        @named @tensor w2[k] := t[k,i,j] * tm[j,i]
        @test w2 == w

        w3 = similar(w)
        @named @tensor w3[k] = t[k,i,j] * tm[i,j];
        @test w3 == w
    end

end
