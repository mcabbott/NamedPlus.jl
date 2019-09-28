#################### RECURSIVE UNWRAPPING ####################

export HasNames, hasnames

using LinearAlgebra

struct HasNames end

"""
    hasnames(A)

Returns `HasNames()` if `A::NamedDimsArray`
or any wrapper around that which defines `parent`,
otherwise `false`.
"""
hasnames(x::NamedDimsArray) = HasNames()

hasnames(x::AbstractArray) = x === parent(x) ? false : hasnames(parent(x))

"""
    thenames(A)

Returns dimension names of `A`.
For any wrapper type which changes the order / number of dimensions,
you will need to define `outmap(x, names) = outernames`.
"""
thenames(x::NamedDimsArray{names}) where {names} = names

function thenames(x::AbstractArray)
    # hasnames(x) === HasNames() || return default_names(x)
    p = parent(x)
    x === p && return default_names(x)
    return outmap(x, thenames(p), :_)
end

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

# nameless(x::AbstractArray) = x === parent(x) ? x : rewraplike(x, parent(x), nameless(parent(x)))
function nameless(x::AbstractArray)
    hasnames(x) === HasNames() || return x
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
    #=
    cnt = 0
    FT = MacroTools.postwalk(AT) do ex
        @show ex
        if ex == PT
            cnt += 1
            return UT
        end
        ex
    end
    cnt == 1 || error("failed to re-wrapping function for this type, please define rewraplike($AT)")
    =#
    FT = Meta.parse(replace(string(AT), string(PT) => string(UT)))
    :( $FT(z) )
end

rewraplike(x::SubArray, y, z) = SubArray(z, x.indices, x.offset1, x.stride1) # untested!

####################
