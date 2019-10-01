#################### RECURSIVE UNWRAPPING ####################

export True, hasnames, getnames

using LinearAlgebra

struct True <: Integer end
Base.promote_rule(::Type{T}, ::Type{True}) where {T<:Number} = T
Base.convert(::Type{T}, ::True) where T<:Number = convert(T, true)

names_doc = """
    names(A::NamedDimsArray) -> Tuple
    names(A, d) -> Symbol

`Base.names` acts as the accessor function.

    hasnames(A)
    getnames(A)
    getnames(A, d)

These work recursively through wrappers.
`hasnames(A) == True()` or `false`.

For any wrapper type which changes the order / number of dimensions,
you will need to define `outmap(x, names) = outernames`.
"""

@doc names_doc
Base.names(x::NamedUnion) = getnames(x)
Base.names(x::NamedUnion, d::Int) = getnames(x, d)

@doc names_doc
hasnames(x::NamedDimsArray) = True()
hasnames(x::AbstractArray) = x === parent(x) ? false : hasnames(parent(x))

@doc names_doc
function getnames(x::AbstractArray)
    # hasnames(x) === True() || return default_names(x)
    p = parent(x)
    x === p && return default_names(x)
    return outmap(x, getnames(p), :_)
end
getnames(x::NamedDimsArray{names}) where {names} = names
getnames(x, d::Int) = d <= ndims(x) ? getnames(x)[d] : :_

default_names(x::AbstractArray) = ntuple(_ -> :_, ndims(x))

"""
    outmap(A, tuple, default)

Maps the names/ranges tuple from `A.parent` to that for `A`.
"""
outmap(A, tup, z) = tup

outmap(::Transpose, x::Tuple{Any}, z) = (z, x...)
outmap(::Transpose, x::Tuple{Any,Any}, z) = reverse(x)

outmap(::Adjoint, x::Tuple{Any}, z) = (z, x...)
outmap(::Adjoint, x::Tuple{Any,Any}, z) = reverse(x)

outmap(::Diagonal, x::Tuple{Any}, z) = (x..., x...)

outmap(::PermutedDimsArray{T,N,P,Q}, x::Tuple, z) where {T,N,P,Q} =
    ntuple(d -> x[P[d]], N)

outmap(::TransmutedDimsArray{T,N,P,Q}, x::Tuple, z) where {T,N,P,Q} =
    ntuple(d -> P[d]==0 ? z : x[P[d]], N)

outmap(::SubArray, x) = (@warn "outmap may behave badly with views!"; x)

"""
    nameless(x)

An attempt at a recursive `unname()` function.
"""
nameless(x::NamedDimsArray) = parent(x)

nameless(x) = x

function nameless(x::AbstractArray)
    hasnames(x) === True() || return x
    p = parent(x)
    p === x && return x
    return rewraplike(x, p, nameless(p))
end


"""
    rewraplike(x, y, z)
    rewraplike(x, parent(x), nameless(parent(x)))

This looks at the type of `x`, replaces the type of `y` with that of `z`,
and then uses that to act on `z`. Hopefully that's the right constructor!
For troublesome wrapper types you may need to overload this.
"""
@generated function rewraplike(x::AT, y::PT, z::UT) where {AT <: AbstractArray{T,N}, PT, UT} where {T,N}
    FT = Meta.parse(replace(string(AT), string(PT) => string(UT)))
    :( $FT(z) )
end

rewraplike(x::SubArray, y, z) = SubArray(z, x.indices, x.offset1, x.stride1) # untested!

####################
