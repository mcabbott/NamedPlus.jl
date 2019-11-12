
#################### ZERO, ONES ####################

zero_doc = """
    zeros(; r=2)
    ones(Int8; r=2, c=3)
    fill(3.14; c=3)

These are piratically overloaded to make `NamedDimsArray`s.
The zero-dimensional methods like `fill(3)` should stil work fine.
See also `rand(Float64; i=10)` etc.
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
"""
diagonal(x) = Diagonal(x)
diagonal(x::NamedDimsArray{L,T,1}) where {L,T} = NamedDimsArray{(L[1],L[1])}(Diagonal(x.data))
diagonal(x::NamedDimsArray{L,T,2}) where {L,T} = NamedDimsArray{L}(Diagonal(x.data))

diagonal(x::AbstractArray, s::Symbol) = diagonal(x, (s,s))
diagonal(x::AbstractArray, tup::Tuple{Symbol, Symbol}) = NamedDimsArray{tup}(Diagonal(x))

####################