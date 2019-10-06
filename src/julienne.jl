using JuliennedArrays

"""
    S = Slices(A, :i, :j)

Returns a sliced view of A's data, in which each slice lies along the dimensions mentioned:
`names(S[1]) == (:i,:j)`, while the other names belong to the outer container,
`getnames(S) ≈ setdiff(getnames(A), (:i,:j))`.

See `Align` for the reverse operation.
"""
function JuliennedArrays.Slices(A::NamedUnion, dims::Symbol...)
    ints = map(d -> NamedDims.dim(getnames(A), d), dims)
    Slices(A, ints...)
end

Base.parent(S::Slices) = S.whole # Piracy! https://github.com/bramtayl/JuliennedArrays.jl/pull/21
# NamedPlus.hasnames(S::Slices) = hasnames(S.whole)
# NamedPlus.hasranges(S::Slices) = hasnames(S.whole)

NamedPlus.nameless(S::Slices) = Slices(nameless(S.whole), S.alongs...) # better to define reconstruct?
NamedPlus.rangeless(S::Slices) = Slices(rangeless(S.whole), S.alongs...)

NamedPlus.outmap(S::Slices, tup::Tuple, def) = out_alongs(S.alongs, tup)

function out_alongs(alongs::Tuple, tup::Tuple)
    rest = out_alongs(Base.tail(alongs), Base.tail(tup))
    first(alongs) === JuliennedArrays.False() ? (first(tup), rest...) : rest
end
out_alongs(alongs::Tuple{}, tup::Tuple{}) = ()

# 7.076 ns (1 allocation: 16 bytes)
# @btime (() -> out_alongs((JuliennedArrays.True(), JuliennedArrays.False()), (:a, :b)))()


"""
    A = Align(S, 1, 2)

This is the usage in `JuliennedArrays`, the numbers are which dimensions of `A`
belong to a single slice like `first(S)`; other dimensions indicate which slice.

    A = Align(S)                # getnames(A) == getnames(getnames(first(S)), getnames(S))
    A = Align(S, :i, :j)        # names(A) == (:i, :j, getnames(S)...)
    A = Align(S, (:i, :j, :k))  # names(A) == (:i, :j, :k)

You can also do three more things.
On full auto, slice dimensions go first, and any dimensions without names will get `:_`.
Next, you can specify names for the slice dimensions.
Finally, you can specify all of the names of the final array,
which fixes the orientation
"""
JuliennedArrays.Align(A::NamedUnion) = Align(A, 1:ndims(first(A))...)
JuliennedArrays.Align(A::AbstractArray{<:NamedUnion}) = Align(A, 1:ndims(first(A))...)

function JuliennedArrays.Align(A::NamedUnion, dims::Tuple{Vararg{Symbol}})
    if hasnames(A) && hasnames(first(A))
        outer = map(d -> NamedDims.dim_noerror(getnames(A), d), dims)
        inner = map(d -> NamedDims.dim_noerror(getnames(first(A)), d), dims)
        # This is not yet correct. Slices can be permuted...
        ints = map(outer, inner) do dout, din
            dout>0 && din==0 && return dout
            dout==0 && din>0 && return din
            dout==0 && din==0 && error("some names not found anywhere!")
            dout>0 && din>0 && error("some names found twice!")
        end
        return Align(A, ints...)

    elseif hasnames(A) # no names on slices, re-wrap outer
        outer = map(d -> NamedDims.dim(getnames(A), d), dims)
        ints = filter(d -> !(d in outer), ntuple(identity, length(dims)))
        return NamedDimsArray(Align(nameless(A), ints...), dims)

    elseif hasnames(first(A))
        ints = map(d -> NamedDims.dim(getnames(first(A)), d), dims)
        outer_names = NamedDims.remaining_dimnames_after_dropping(dims, ints)
        return Align(NamedDimsArray(A, outer_names), ints...)

    else
        return NamedDimsArray(Align(A, (1:ndims(first(A)))...), dims)
    end
end

NamedPlus.hasnames(A::Align) = hasnames(A.slices) || hasnames(first(A.slices))
NamedPlus.hasranges(A::Align) = hasranges(A.slices) || hasranges(first(A.slices))

function NamedPlus.nameless(A::Align)
    outer = hasnames(A.slices) ? nameless(A.slices) : A.slices
    inner = hasnames(first(outer)) ? map(nameless, outer) : outer
    Align(inner, A.alongs...)
end

NamedPlus.outmap(A::Align, tup::Tuple, def) = out_alongs(A.alongs, tup)

NamedPlus.getnames(A::Align) = align_names(A.alongs::Tuple, getnames(A.slices), getnames(first(A.slices)))

function align_names(alongs::Tuple, outer, inner)
    if first(alongs) === JuliennedArrays.True()
        (first(inner), align_names(Base.tail(alongs), outer, Base.tail(inner))...)
    else
        (first(outer), align_names(Base.tail(alongs), Base.tail(outer), inner)...)
    end
end
align_names(alongs::Tuple{}, outer, inner) = ()
