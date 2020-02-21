#################### ZERO, ONES ####################

_zero_doc = """
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

julia> using OffsetArrays

julia> zeros(_=1, col=11:17)
1×7 NamedDimsArray(OffsetArray(::Array{Float64,2}, 1:1, 11:17), (:_, :col)):
      → :col
↓ :_  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
@doc _zero_doc
Base.zeros(; kw...) = zeros(Float64; kw...)

function Base.zeros(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(zeros(T, kw.data...), kw.itr)
    else
        fill(zero(T), ())
    end
end

@doc _zero_doc
Base.ones(; kw...) = ones(Float64; kw...)

function Base.ones(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(ones(T, kw.data...), kw.itr)
    else
        fill(one(T), ())
    end
end

@doc _zero_doc
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

# @doc _zero_doc
# Base.rand(; kw...) = rand(Float64; kw...)
# @doc _zero_doc
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
        rand(NamedPlus.RNG(), T) # @btime rand() unchanged
    end
end
function Base.randn(T::Type{Tn}; kw...) where {Tn<:Number}
    if length(kw) >= 1
        NamedDimsArray(randn(T, kw.data...), kw.itr)
    else
        randn(NamedPlus.RNG(), T)
    end
end

"""
    range(; i=10) == NamedDimsArray(1:10, :i)

Keyword piracy to make ranges.
"""
function Base.range(; kw...)
    if length(kw.itr) == 1
        NamedDimsArray(NamedPlus._ensure_range(kw.data[1]), kw.itr)
    elseif length(kw.itr) >= 2
        data = NamedPlus.outer(NamedPlus._ensure_range.(Tuple(kw.data))...)
        NamedDimsArray(data, kw.itr)
    else
        error("range() with no arguments still doesn't mean anything")
    end
end

#################### PRIMES ####################

"""
    :x' == :x′

`adjoint(::Symbol)` adds unicode prime `′` to the end.
"""
Base.adjoint(s::Symbol) = NamedPlus.prime(s)

####################
