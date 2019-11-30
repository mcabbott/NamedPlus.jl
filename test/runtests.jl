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

    # comprehensions
    @test (@named [i^2 for i in 1:3]) isa NamedDimsArray
    @test names(@named [i/j for i in 1:3, j in 1:4]) == (:i, :j)

    @test names(@named [x^2 for x in 1:2:10]) == (:x,)
    @test names(@named [x^i for x in 1:2:10, i in 1:3]) == (:x, :i)
    @test_skip ranges(@named [x^2 for x in 1:2:10]) == (1:2:10,)
    @test_skip ranges(@named [x^i for x in 1:2:10, i in 1:3]) == (1:2:10, 1:3)

    # printing
    m4 = named(Int8[1 2; 3 4], :i, :j)
    @test repr(m4) == "named(Int8[1 2; 3 4], :i, :j)"

end
@testset "align" begin

    v1 = align(v, (:j, :k))
    @test v1 == v
    v1 = align(v', (:j, :k))
    @test v1 == v

    v2 = align(v, (:x, :j, :y))
    @test v2 == v'
    v2 = align(v', (:x, :j, :y))
    @test v2 == v'

    m1 = align(m, (:i, :j))
    @test names(m1) == (:i, :j)
    @test size(m1) == (2,3)
    m1 = align(m, (:i, :j, :x, :y))
    @test names(m1) == (:i, :j)
    @test size(m1) == (2,3)

    m2 = align(m, (:j,:i))
    @test names(m2) == (:j, :i)
    @test size(m2) == (3,2)
    m2 = align(m, (:j,:i, :x, :y))
    @test names(m2) == (:j, :i)
    @test size(m2) == (3,2)

    m3 = align(m, (:i, :z, :j))
    @test names(m3) == (:i, :_, :j)
    @test size(m3) == (2,1,3)
    m3 = align(m, (:i, :x, :j, :y, :z))
    @test names(m3) == (:i, :_, :j)
    @test size(m3) == (2,1,3)

    m4 = align(m, (:w, :j, :z, :i))
    @test names(m4) == (:_, :j, :_, :i)
    @test size(m4) == (1,3,1,2)
    m4 = align(m, (:w, :j, :z, :i, :x, :y))
    @test names(m4) == (:_, :j, :_, :i)
    @test size(m4) == (1,3,1,2)

end
@testset "aligned reduction" begin

    @test_throws DimensionMismatch sum!(v, m)
    @test sum!ᵃ(v, m) == sum!(v, m')
    @test prod!ᵃ(m', t) == prod!(m, t)'

end
#=
# broken without TransmuteDims master?
@testset "broadcasting by name" begin

    @test names(m ./ v') == (:i, :j)
    @test_throws DimensionMismatch m ./ v

    @test names(@named w{i,k,j} = t .+ m ./ v) == (:i, :k, :j)
    @named z{i,j,k} = t .+ m ./ v
    @test z == t .+ m ./ v'

end
=#
@testset "rename & prime" begin

    @test prime(:z) == :z′
    @test names(prime(m, first)) == (:i′, :j)
    @test names(prime(m, 2)) == (:i, :j′)

    @test names(rename(m, :j => :k)) == (:i, :k)
    @test names(rename(m, :j => :k, :i => :j)) == (:j, :k)
    @test names(rename(m, :j => :k, :k => :l)) == (:i, :l)
    @test names(rename(m, (:a, :b))) == (:a, :b)

    using NamedPlus: _prime

    @test (@inferred (() -> prime(:a))() ;true)
    @test (@inferred (() -> _prime((:i,:j,:k), Val(1)))() ;true)
    @test 0 == @allocated (() -> prime(:a))()
    @test 0 == @allocated (() -> _prime((:i,:j,:k), Val(1)))()

end
@testset "split & join" begin

    @test names(join(t, (:i,:j) => :ij)) == (:ij, :k)
    t1 = join(t, :i,:k)  # these are not neighbours
    t1′ = join(t, :k,:i) # reverse order
    @test names(t1) == (:j, :iᵡk)
    @test names(t1′) == (:j, :kᵡi)

    @test size(split(m, :i => (:i1, :i2), (1,2))) == (1, 2, 3)

    @test t == split(join(t, (:i,:j) => :ij), :ij => (:i,:j), (2,3))

    t2 = split(t1, :iᵡk => (:i,:k), (2,4));
    @test names(t2) == (:j, :i, :k)
    @test size(t2) == (3, 2, 4)
    t2′ = split(t1′, :kᵡi => (k=4, i=2));
    t2′′ = split(t1′, :kᵡi => (i=2, k=4));
    @test t2[j=1] == transpose(t2′[j=1])
    @test_broken t2[j=1] == t2′′[j=1]

    t2[1,1,1] = 99
    @test t1[1,1,1] == 99
    @test t[1,1,1] != 99

    using NamedPlus: _join, _split

    @test (@inferred (() -> _join(:i, :j))() ;true)
    @test (@inferred (() -> _split(_join(:i, :j)))() ;true)
    @test 0 == @allocated (() -> _join(:i, :j))()
    @test 0 == @allocated (() -> _split(_join(:i, :j)))()

end
@testset "vec, dropdims" begin

    @test names(vec(v)) ==  (:j,)
    @test names(vec(m)) ==  (:iᵡj,)
    @test names(vec(t)) ==  (:_,)

    @test size(dropdims(NamedDimsArray(rand(2,1,1), (:a, :_, :_)))) == (2,)
    @test names(dropdims(named(ones(2,2,1,1,2), :a, :b, :_, :_, :c))) == (:a, :b, :c)

    # Test my pirate methods for _dropdims(::Transpose, 1) etc.
    r3 = rand(3)
    @test dropdims(r3', dims=1) isa Array
    @test dropdims(r3', dims=1) === r3
    @test dropdims(r3 |> transpose, dims=1) === r3
    @test dropdims(rand(1)', dims=2) isa Base.ReshapedArray # unchanged
    @test names(dropdims(v')) == (:j,) # with default dims=:_

end
@testset "named int" begin

    ni, nj = size(m)
    @test ni isa NamedInt

    # Creators
    @test names(zeros(ni, nj)) == (:i, :j)
    @test names(ones(ni, nj)) == (:i, :j)
    @test names(rand(ni, nj)) == (:i, :j)
    @test names(randn(ni, nj)) == (:i, :j)

    @test names(zeros(Int, ni, nj)) == (:i, :j)
    @test names(ones(Float32, ni, nj)) == (:i, :j)
    @test names(rand(Int8, ni, nj)) == (:i, :j)
    @test names(randn(Float64, ni, nj)) == (:i, :j)

    # Ranges
    @test names(1:ni) == (:i,)
    @test names([x^i for x in 1:nj, i in 1:ni]) == (:j, :i)

    # reshape
    @test names(reshape(rand(6), ni, nj)) == (:i, :j)
    @test names(reshape(rand(1,6,1), nj, :)) == (:j, :_)

end
@testset "base piracy" begin

    # Base behaviour
    @test ones() isa Array{Float64, 0}
    @test zeros() isa Array{Float64, 0}
    @test fill(3.14) isa Array{Float64, 0}
    @test ones(3) isa Array{Float64, 1}
    @test zeros(3,4) isa Array{Float64, 2}
    @test fill(3.14, 3,4,5) isa Array{Float64, 3}
    @test range(3, stop=4) === 3:4

    @test rand() isa Float64
    @test randn() isa Float64
    @test rand(Float64) isa Float64
    @test randn(Float64) isa Float64
    @test rand(1,2) isa Array{Float64, 2}
    @test randn(1,2) isa Array{Float64, 2}
    @test rand(Float64,1,2) isa Array{Float64, 2}
    @test randn(Float64,1,2) isa Array{Float64, 2}

    # Overloads
    @test names(ones(i=3)) == (:i,)
    @test names(ones(Int, i=3)) == (:i,)
    @test names(zeros(i=3, j=4)) == (:i,:j)
    @test names(zeros(Int, i=3, j=4)) == (:i,:j)
    @test names(fill(3.14, i=3, j=4, k=5)) == (:i,:j,:k)

    @test names(rand(Float64, i=3)) == (:i,)
    @test names(randn(Float64, i=3, j=4)) == (:i,:j)
    @test eltype(rand(Int8, i=3)) == Int8

    @test names(range(i=10)) == (:i,)
    @test parent(range(i=10)) isa Base.OneTo
    @test names(range(i=3:4)) == (:i,)

end
#=
# These require https://github.com/invenia/NamedDims.jl/pull/79
@testset "matrix mult" begin

    ab = rand(Int8, a=2, b=2)
    bc = rand(Int8, b=2, c=2)
    ca = rand(Int8, c=2, a=2)
    aa = rename(bc, :b => :a, :c => :a)

    @test mul(ab, bc) == ab * bc
    @test mul(ab', bc') == ab * bc
    @test mul(ab, ca) == ab' * ca'
    @test_throws Exception mul(ab, ab')
    @test_throws Exception mul(ab, aa)

    @test mul(ab, ab, :a) == ab' * ab
    @test mul(ab, ab, :b) == ab * ab'
    @test_throws Exception mul(ab, aa, :a)

    a = rand(Int8, a=2)

    mul(a, a) == a' * a
    mul(a, ca) == ca * a
    mul(a, ab) == ab' * a
    @test_throws Exception mul(a, bc)
    @test_throws Exception mul(a, aa)

    @test ab *ᵃ bc == ab * bc

end
=#

using LinearAlgebra, TensorOperations, TransmuteDims, EllipsisNotation

@testset "named and .." begin

    names(named(ones(1,1,1,1), ..)) == (:_, :_, :_, :_)
    names(named(ones(1,1,1,1), :a, ..)) == (:a, :_, :_, :_)
    names(named(ones(1,1,1,1), :a, :b, ..)) == (:a, :b, :_, :_)
    names(named(ones(1,1,1,1), :a, :b, :c, :d, ..)) == (:a, :b, :c, :d)

    names(named(ones(1,1,1,1), .., :z)) == (:_, :_, :_, :z)
    names(named(ones(1,1,1,1), .., :y, :z)) == (:_, :_, :y, :z)
    names(named(ones(1,1,1,1), .., :w, :x, :y, :z)) == (:w, :x, :y, :z)

    names(named(ones(1,1,1,1), :a, .., :z)) == (:a, :_, :_, :z)
    names(named(ones(1,1,1,1), :a, :b, .., :z)) == (:a, :b, :_, :z)
    names(named(ones(1,1,1,1), :a, .., :y, :z)) == (:a, :_, :y, :z)
    names(named(ones(1,1,1,1), :a, :b, .., :y, :z)) == (:a, :b, :y, :z)

    @test_throws Exception named(ones(1,1,1), :a, :b, .., ..)

end
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

    s = rand(Int8, 1,2,3)
    tns = TransmutedDimsArray(NamedDimsArray(s,(:a, :b, :c)), (2,3,0,1))
    @test canonise(tns) == s
    @test names(canonise(tns)) == (:a, :b, :c) # easy!
    nts = NamedDimsArray(TransmutedDimsArray(s, (2,3,0,1)), (:b, :c, :_, :a))
    @test canonise(nts) == s
    @test names(canonise(nts)) == (:a, :b, :c) # permutes names

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
#=
# These require https://github.com/invenia/NamedDims.jl/pull/79
@testset "contract matrices" begin

    ab = rand(Int8, a=2, b=2)
    bc = rand(Int8, b=2, c=2)
    ca = rand(Int8, c=2, a=2)
    aa = rename(bc, :b => :a, :c => :a)

    @test contract(ab, bc) == ab * bc
    @test contract(ab', bc') == ab * bc
    @test contract(ab, ca) == ab' * ca'
    contract(ab, ab')
    @test_throws Exception contract(ab, aa)

    @test contract(ab, ab, :a) == ab' * ab
    @test contract(ab, ab, :b) == ab * ab'
    @test_throws Exception contract(ab, aa, :a)

    a = rand(Int8, a=2)

    contract(a, a) == a' * a
    contract(a, ca) == ca * a
    contract(a, ab) == ab' * a
    contract(a, bc) == outer(a, bc)
    @test_throws Exception contract(a, aa)


end
=#

#=
@info "Done with own tests, now running those of NamedDims.jl to check that rampant piracy hasn't sunk anything important"
# but now two reshape test will fail.
@testset "test from NamedDims" begin

    folder = dirname(pathof(NamedDims))
    file = normpath(joinpath(folder, "..", "test", "runtests.jl"))
    try
        include(file)
    catch e
        @show e
    end

end
=#
