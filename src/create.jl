
#################### NAMED ####################

using EllipsisNotation
const Dots = EllipsisNotation.Ellipsis # now identical to IntervalSets's.

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
named(A::AbstractArray, B::NamedUnion) = named(A, getnames(B)...)

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
    if A isa NamedDimsArray
        refine_names(A, final) # from v0.2.16
    else
        NamedDimsArray(A, final)
    end
end

#################### PIRACY ####################

"""
    @pirate Base

This will define all sorts of convenient methods,
like `ones(i=2)` and `rand(Float64, j=3)`.
"""
macro pirate(exs...=(:Base,))
    out = quote end

    if :Base in exs
        str = read(joinpath(@__DIR__, "pirate.jl"), String)
        n = 1
        while true
            expr, n = Meta.parse(str, n; greedy=true)
            expr === nothing && break
            push!(out.args, expr)
        end
        push!(out.args, nothing)
    end

#     @pirate NamedDims
# This causes `size(nda)` to return a tuple of `NamedInt`s,
# for propagating names through some code.
    # if :NamedDims in exs
    #     str = read(joinpath(@__DIR__, "int.jl"), String)
    #     n = 1
    #     while true
    #         expr, n = Meta.parse(str, n; greedy=true)
    #         expr === nothing && break
    #         push!(out.args, expr)
    #     end
    #     push!(out.args, nothing)
    # end

    out
end

# export NamedInt, ᵅ

# Random.default_rng()::AbstractRNG
using Random
if VERSION < v"1.3-"
    RNG() = Random.GLOBAL_RNG
else
    const RNG = Random.default_rng
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
        dims += ndims(y)
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
