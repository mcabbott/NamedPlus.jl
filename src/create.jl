
#################### NAMED ####################

import IntervalSets, EllipsisNotation
const Dots = Union{typeof(IntervalSets.:(..)), EllipsisNotation.Ellipsis}

"""
    A′ = named(A, names...)

This differs from `NamedDimsArray(A, names)` in that
it allows `..` to fill in as many wildcards as needed.
(And it saves typing nine letters and two brackets.)

To use `..` you need either IntervalSets or EllipsisNotation.
```
julia> named(fill(π, 1,1), :α, :β)
1×1 NamedDimsArray(::Array{Irrational{:π},2}, (:α, :β)):
      → :β
↓ :α  π

julia> using EllipsisNotation

julia> named(ones(2,3,1,1), :a, :b, .., :z)
2×3×1×1 NamedDimsArray(::Array{Float64,4}, (:a, :b, :_, :z)):
[:, :, 1, 1] =
      → :b
↓ :a  1.0  1.0  1.0
      1.0  1.0  1.0
```
"""
named(A::AbstractArray, names::Tuple) = named(A, names...)
named(A::AbstractArray, B::NamedUnion) = named(A, names(B))

function named(A::AbstractArray, names::Union{Symbol, Dots}...)
    names isa Tuple{Vararg{Symbol}} && return NamedDimsArray(A, names)

    found = 0
    tup = ntuple(length(names)) do d
        n = names[d]
        n isa Symbol && return 0
        found += 1
        found ==1 && return d
        error("can't have two ..!")
    end
    early = sum(tup) - 1
    late = ndims(A) - length(names) + sum(tup) + 1
    off = late - sum(tup) - 1

    final = ntuple(ndims(A)) do d
        d <= early && return names[d]
        d >= late && return names[d - off]
        return :_
    end

    late > early || @warn "not all names in $names will be used in output $final"
    NamedDimsArray(A, final)
end

#################### ZERO, ONES ####################

zero_doc = """
    zeros(; r=2)
    ones(Int8; r=2, c=3)
    fill(3.14; c=3)

These are piratically overloaded to make `NamedDimsArray`s.
The zero-dimensional methods like `fill(3)` should stil work fine.
See also `rand(Float64; i=10)` and `range(; i=10)`.
```
julia> fill(π, α=1, β=1)
1×1 NamedDimsArray(::Array{Irrational{:π},2}, (:α, :β)):
      → :β
↓ :α  π

julia> zeros(ComplexF64, r=2, c=3)
2×3 NamedDimsArray(::Array{Complex{Float64},2}, (:r, :c)):
      → :c
↓ :r  0.0+0.0im  0.0+0.0im  0.0+0.0im
      0.0+0.0im  0.0+0.0im  0.0+0.0im
```
"""
@doc zero_doc
Base.zeros(; kw...) = zeros(Float64; kw...)

function Base.zeros(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(zeros(T, kw.data...), kw.itr)
    else
        fill(zero(T), ())
    end
end

@doc zero_doc
Base.ones(; kw...) = ones(Float64; kw...)

function Base.ones(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(ones(T, kw.data...), kw.itr)
    else
        fill(one(T), ())
    end
end

@doc zero_doc
function Base.fill(v; kw...)
    if length(kw) >= 1
        NamedDimsArray(fill(v, Tuple(kw.data)), kw.itr)
    else
        fill(v, ())
    end
end

#=# Note that rand(; kw...) is worse piracy than ones(; kw...)

julia> @which zeros()
zeros(dims::Union{Integer, AbstractUnitRange}...) in Base at array.jl:454

julia> @which fill(1)
fill(v, dims::Union{Integer, AbstractUnitRange}...) in Base at array.jl:407

julia> @which rand()
rand() in Random at /Users/julia/buildbot/worker/package_macos64/build/usr/share/julia/stdlib/v1.3/Random/src/Random.jl:256

julia> @which rand(Random.default_rng())
rand(rng::AbstractRNG) in Random at
=#

# Random.default_rng()::AbstractRNG
using Random
if VERSION < v"1.3-"
    RNG() = Random.GLOBAL_RNG
else
    const RNG = Random.default_rng
end
# @doc zero_doc
# Base.rand(; kw...) = rand(Float64; kw...)
# @doc zero_doc
# Base.randn(; kw...) = rand(Float64; kw...)

"""
    rand(Float32; c=3, d=4)
    randn(Float64; c=3, d=4)

Keyword overload for `rand` needs a type, otherwise too piratical.
See also `zeros(; c=3, d=4)`, which works with or without the type.
```
julia> rand(Int8, a=2, b=4)
2×4 NamedDimsArray(::Array{Int8,2}, (:a, :b)):
      → :b
↓ :a   32   21   87  105
      -42  -32  -35   33
```
"""
function Base.rand(T::Type{Tn}; kw...) where {Tn<:Number}
    if length(kw) >= 1
        NamedDimsArray(rand(T, kw.data...), kw.itr)
    else
        rand(RNG(), T) # @btime rand() unchanged
    end
end
function Base.randn(T::Type{Tn}; kw...) where {Tn<:Number}
    if length(kw) >= 1
        NamedDimsArray(randn(T, kw.data...), kw.itr)
    else
        randn(RNG(), T)
    end
end

"""
    range(; i=10) == NamedDimsArray(1:10, :i)

Keyword piracy to make ranges.
"""
function Base.range(; kw...)
    if length(kw.itr) == 1
        NamedDimsArray(_ensure_range(kw.data[1]), kw.itr)
    elseif length(kw.itr) >= 2
        data = outer(_ensure_range.(Tuple(kw.data))...)
        NamedDimsArray(data, kw.itr)
    else
        error("range() with no arguments still doesn't mean anything")
    end
end

_ensure_range(n::Integer) = Base.OneTo(n)
_ensure_range(r::AbstractRange) = r
_ensure_range(r) = error("not sure what to do with $r, expected an integer or a range")

#################### OUTER ####################

"""
    outer(x, y) = outer(*, x, y)

This inserts enough new dimensions, then broadcasts `x .* y′`.
```
julia> outer(ones(β=2), rand(Int8, δ=4))
2×4 NamedDimsArray(::Array{Float64,2}, (:β, :δ)):
      → :δ
↓ :β  116.0  61.0  106.0  -77.0
      116.0  61.0  106.0  -77.0
```
"""
outer(xs::AbstractArray...) = outer(*, xs...)

function outer(f::Function, x::AbstractArray, ys::AbstractArray...)
    dims = ndims(x)
    views = map(ys) do y
        newaxes = ntuple(_->newaxis, dims)
        colons = ntuple(_->(:), ndims(y))
        view(y, newaxes..., colons...)
    end
    Broadcast.broadcast(f, x, views...)
end

const newaxis = [CartesianIndex()]

#################### DIAGONAL ####################

"""
    diagonal(m)

`diagonal` is to `Diagonal` about as `transpose` is to `Transpose`.
On an ordinary Vector there is no distinction, but on a NamedDimsArray it returns
`NamedDimsArray{T,2,Diagonal}` instead of `Diagonal{T,NamedDimsArray}`.
This has two independent names, by default the same as that of the vector,
but writing `diagonal(m, names)` they can be set.
```
julia> diagonal(ones(i=3))
3×3 NamedDimsArray(::Diagonal{Float64,Array{Float64,1}}, (:i, :i)):
      → :i
↓ :i  1.0   ⋅    ⋅
       ⋅   1.0   ⋅
       ⋅    ⋅   1.0

julia> diagonal(11:13, :i, :i′)
3×3 NamedDimsArray(::Diagonal{Int64,UnitRange{Int64}}, (:i, :i′)):
      → :i′
↓ :i  11   ⋅   ⋅
       ⋅  12   ⋅
       ⋅   ⋅  13
```
"""
diagonal(x) = Diagonal(x)
diagonal(x::NamedDimsArray{L,T,1}) where {L,T} = NamedDimsArray{(L[1],L[1])}(Diagonal(parent(x)))
diagonal(x::NamedDimsArray{L,T,2}) where {L,T} = NamedDimsArray{L}(Diagonal(parent(x)))

diagonal(x::AbstractArray, s::Symbol) = diagonal(nameless(x), (s,s))
diagonal(x::AbstractArray, s1::Symbol, s2::Symbol) = diagonal(nameless(x), (s1,s2))
diagonal(x::AbstractArray, tup::Tuple{Symbol, Symbol}) = NamedDimsArray{tup}(Diagonal(nameless(x)))

####################
