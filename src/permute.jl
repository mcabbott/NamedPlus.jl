#################### UNWRAPPING ####################
## https://github.com/invenia/NamedDims.jl/issues/65

"""
    unname(A::NamedDimsArray, names) -> AbstractArray

Returns the parent array if the given names match those of `A`,
otherwise a `transpose` or `PermutedDimsArray` view of the parent data.
Like `parent(permutedims(A, names))` but making a view not a copy.

This is now simply `parent(permutenames(A, names))`,
and thus it allows `names` longer than `ndims(A)`, which will insert trivial dimensions.
"""
NamedDims.unname(A::NamedUnion, names::NTuple{N, Symbol}) where {N} =
    parent(permutenames(A, names))


#################### PERMUTENAMES ####################

export permutenames, permutenames2

"""
    permutenames(A, names)

This is a bit like `permutedims`, but does not copy the data to a new array in the given order,
and instead wraps it in `Transpose` or `PemutedDimsArray` as nedded.

If `A` has fewer dimensions than `names`, then trivial dimensions are inserted
using a `TransmutedDimsArray`.

Note that `canonise(A)` unwraps `Diagonal{...,Vector}` and `Transpose{...,Vector}`
to have just one index.
"""
function permutenames(A::NamedUnion, target::Tuple{Vararg{Symbol}}; lazy::Bool=true)
    B = canonise(A)
    T = eltype(A)

    namesB = getnames(B)
    perm = map(n -> NamedDims.dim_noerror(namesB, n), target)

    if perm == ntuple(identity, ndims(B))
        return B

    elseif (perm == (2,1) || perm == (0,1)) && T <: Number
        return transpose(B)

    elseif 0 in perm
        L = map(n -> Base.sym_in(n, namesB) ? n : :_, target)
        out = NamedDimsArray{L}(Transmute{perm}(nameless(B)))
        return out

    else
        C = PermutedDimsArray(nameless(B), perm)
        return NamedDimsArray{target}(C)
    end
end



#################### PERMUTE ####################

function Base.PermutedDimsArray(nda::NamedUnion, perm::Tuple{Vararg{Symbol}})
    PermutedDimsArray(nda, dim(getnames(nda), perm))
end

function TransmuteDims.TransmutedDimsArray(nda::NamedUnion, perm::Tuple{Vararg{Symbol}})
    list = getnames(nda)
    prime = map(i -> NamedDims.dim_noerror(list,i), perm)
    TransmutedDimsArray(nda, prime)
end

# this is enough to make Transmute{sym} work, I think:
# No, A here is a type not an instance, getnames doesn't work :(
#=
@inline function TransmuteDims.sanitise_zero(perm::NTuple{N,Symbol}, A) where {N}
    # list = getnames(A)
    list = NamedDims.names(A)
    map(i -> NamedDims.dim_noerror(list,i), perm)
end
=#

# This works but may not be fast yet:

function TransmuteDims.Transmute{perm}(data::A) where {A<:NamedUnion, perm}
    M = ndims(A)
    T = eltype(A)
    if perm isa Tuple{Vararg{Symbol}}
        list = getnames(data)
        prime = map(i -> NamedDims.dim_noerror(list,i), perm)
        perm_plus = TransmuteDims.sanitise_zero(prime, data)
    else
        perm_plus = TransmuteDims.sanitise_zero(perm, data)
    end
    real_perm = TransmuteDims.filter(!iszero, perm_plus)
    length(real_perm) == M && isperm(real_perm) || throw(ArgumentError(
        string(real_perm, " is not a valid permutation of dimensions 1:", M,
            ". Obtained by filtering input ",perm)))

    N = length(perm_plus)
    iperm = TransmuteDims.invperm_zero(perm_plus, M)
    L = issorted(real_perm)

    # :( TransmutedDimsArray{$T,$N,$perm_plus,$iperm,$A,$L}(data) )
    TransmutedDimsArray{T,N,perm_plus,iperm,A,L}(data)
end

#################### CANONICALISE ####################

export canonise

"""
    A′ = canonise(A::NamedDimsArray)

Re-arranges the index names of `A` to canonical order,
meaning that they now match the underlying storage,
removing lazy re-orderings such as `Transpose`.

This should not affect any operations which work on the index names,
but will confuse anything working on index positions.

`NamedDimsArray{L,T,2,Diagonal}` is unwrapped only if it has two equal names,
which `Diagonal{T,NamedVec}` always has, and thus is always unwrapped to a vector,
"""
canonise(x) = begin
    # @info "nothing to canonicalise?" typeof(x)
    x
end

# diagonal / Diagonal
canonise(x::Diagonal{T,<:NamedDimsArray{L,T,1}}) where {L,T} = x.diag
canon_names(::Type{Diagonal{T,<:NamedDimsArray{L,T,1}}}) where {L,T} = L

canonise(x::NamedDimsArray{L,T,2,<:Diagonal{T,<:AbstractArray{T,1}}}) where {L,T<:Number} =
    L[1] === L[2] ? NamedDimsArray{(L[1],)}(x.data.diag) : x
canon_names(::Type{NamedDimsArray{L,T,2,<:Diagonal{T,<:AbstractArray{T,1}}}}) where {L,T<:Number} =
    L[1] === L[2] ? (L[1],) : L # TODO make canon_names for all the rest? automate?
                                # now you could use outmap() for this.

# PermutedDimsArray
canonise(x::PermutedDimsArray{T,N,P,Q,NamedDimsArray{L,T,N,S}}) where {L,T,N,S,P,Q} =
    NamedDimsArray{L}(x.parent.data)

# transpose / Transpose of a matrix (of numbers, to avoid recursion)
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,2}}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(x.data.parent)

canonise(x::Transpose{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(x.parent.data)

# transpose / Transpose of a vector: drop :_ dimension
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,1}}}) where {L,T<:Number} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(x.data.parent) : x

canonise(x::Transpose{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Number} = x.parent

# you could add Transpose of an array of Transposes, but maybe don't.

# same for Adjoint but only on reals
canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,2}}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(x.data.parent)

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(x.parent.data)

canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,1}}}) where {L,T<:Real} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(x.data.parent) : x

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Real} = x.parent

####################
