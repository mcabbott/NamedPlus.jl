using Test, NamedPlus

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
    @test Base.names(v2) == (:j,)
    @test Base.names(t2,2) == :j

end
@testset "similar" begin

    @test size(similar(t, :k)) == (4,)
    @test eltype(similar(t, Int, :k)) == Int
    @test size(similar(m, z, :i, :z)) == (2,26)
    @test names(similar(v, m, t, (:i, :k))) == (:i, :k)

end
@testset "unname with names" begin

    @test unname(m, (:j, :i)) === transpose(parent(m))

    @test unname(t, (:i,:j,:k)) === parent(t)
    @test unname(t, (:i,:k,:j)) === PermutedDimsArray(parent(t), (1,3,2))

    @test names(permutenames(m, (:i, :k, :j))) == (:i, :_, :j)
    @test names(permutenames(v, (:i, :j, :k))) == (:_, :j, :_)

end

# broken without TransmuteDims master?
@testset "broadcasting by name" begin

    @test names(m ./ v') == (:i, :j)
    @test_throws DimensionMismatch m ./ v

    @test names(@named w{i,k,j} = t .+ m ./ v) == (:i, :k, :j)
    @named z{i,j,k} = t .+ m ./ v
    @test z == t .+ m ./ v'

end

@testset "rename & prime" begin

    @test prime(:z) == :z′
    @test names(prime(m, first)) == (:i′, :j)
    @test names(prime(m, 2)) == (:i, :j′)

    @test names(rename(m, :j => :k)) == (:i, :k)
    @test names(rename(m, (:a, :b))) == (:a, :b)

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
    @test names(d) == (:j, :j)
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
    @test Base.names(p) == (:k, :i, :j)
    @test summary(p) == "k≤4 × i≤2 × j≤3 PermutedDimsArray{Float64,3,(3, 1, 2),(2, 3, 1),NamedDimsArray{(:i, :j, :k),Float64,3,Array{Float64,3}}}"
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

using OffsetArrays

@testset "ranges" begin

    R = RangeWrap(rand(1:99, 3,4), (['a', 'b', 'c'], 10:10:40))
    N = Wrap(rand(1:99, 3,4), obs = ['a', 'b', 'c'], iter = 10:10:40) # combined constructor

    # ===== access
    @test R('c', 40) == R[3, 4]
    @test R('b') == R[2,:]

    @test N(obs='a', iter=40) == N[obs=1, iter=4]
    @test N(obs='a') == N('a') == N[1,:]

    # ===== recursion
    @test getnames(Transpose(N)) == (:iter, :obs)
    @test getranges(Transpose(N)) == (10:10:40, ['a', 'b', 'c'])

    @test nameless(Transpose(N)) isa Transpose{Int, <:RangeWrap}

    N2 = NamedDimsArray(nameless(N), getnames(N))
    @test N2 isa NamedDimsArray{(:obs, :iter),Int,2,<:RangeWrap}
    if VERSION >= v"1.3-rc2"
        @test N2(obs='a', iter=40) == N2[obs=1, iter=4]
    end

    # ===== selectors
    @test N(iter=Near(12.5)) == N[:,1]
    @test_broken N(iter=Between(7,23)) == N[:,1:2]

    @test R('a', Index[2]) == R[1,2]

    # ===== mutation
    V = Wrap([3,5,7,11], μ=10:10:40)
    @test ranges(push!(V, 13)) == 10:10:50

    # ===== offset
    o = OffsetArray(rand(1:99, 5), -2:2)
    w = Wrap(o, i='a':'e')
    @test w[i=-2] == w('a')

end

@testset "comprehensions" begin

    @test names(@named [x^2 for x in 1:2]) == (:x,)
    @test names(@named [x/y for x in 1:2, y in 1:3]) == (:x,:y)

    @test getranges(@named [x^2 for x in 1:2:10]) == (1:2:10,)

end
