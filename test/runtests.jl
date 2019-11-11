using Test, NamedPlus
using NamedDims: names

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
    @test names(v2) == (:j,)
    @test_skip names(t2,2) == :j

    @test (@named [i^2 for i in 1:3]) isa NamedDimsArray
    @test names(@named [i/j for i in 1:3, j in 1:4]) == (:i, :j)

    @test_skip ranges(@named [i^2 for i in 1:2:5]) == (1:2:5,)

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
    @test names(t1) == (:j, :i_k)

    @test size(split(m, :i => (:i1, :i2), (1,2))) == (1, 2, 3)

    @test t == split(join(t, (:i,:j) => :ij), :ij => (:i,:j), (2,3))

    t2 = split(t1, :i_k => (:i,:k), (2,4));
    @test names(t2) == (:j, :i, :k)
    @test size(t2) == (3, 2, 4)

    t2[1,1,1] = 99
    @test t1[1,1,1] == 99
    @test t[1,1,1] != 99

    using NamedPlus: _join, _split

    @test (@inferred (() -> _join(:i, :j))() ;true)
    @test (@inferred (() -> _split(_join(:i, :j)))() ;true)

end

using LinearAlgebra, TensorOperations

@testset "wrapper types" begin

    d = Diagonal(v)
    @test getnames(d) == (:j, :j)
    @test typeof(nameless(d)) == Diagonal{Float64,Array{Float64,1}}

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
    @test t === canonise(p)

end
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
