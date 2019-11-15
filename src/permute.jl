
#################### LAZY ALIGN ####################

"""
    A′ = align(A, names)
    A′ = align(A, B) = align(A, names(B))

This is a lazy generalised `permutedims`. It will do nothing if the names match up,
or insert `Transpose` etc. if they need to be permuted.
Instead of providing `names`, you can also provide another array `B` whose named should be used.

If there are extra dimensions in `names` not found in `A`, then trivial dimensions are inserted.
And if there are dimensions of `A` not found in `names`, these are put at the end, so that the
first few dimensions match `names`. This ensures that the result can be broadcast with `B`.

It should error if wildcards in either set of names make this process ambiguous.

Before performing these steps, is calls `canonise(A)` which unwraps any lazy wrappers,
so that for instance the trivial dimension of `transpose(`named vector`)` is dropped.
"""
align(A::NamedUnion, ::NamedDimsArray{L}) where {L} = align(A, L)

function align(A::NamedUnion, target::Tuple{Vararg{Symbol}})
    B = canonise(A)
    namesB = getnames(B)

    allunique(namesB) || error("input must have unique names, got $namesB" *
        (getnames(A)==getnames(B) ? "" : " after canonicalisation"))
    # Base.sym_in(:_, target) && Base.sym_in(:_, namesB) && error(
    #     "if the input has wildcards, and the target has either wildcards or new symbols, then result is always ambigous. Got $namesB and $target") # wildcards in B get pushed to end anyway, and then it comes here
    Base.sym_in(:_, namesB) && error("input must not have wildcards, got $namesB" *
        (getnames(A)==getnames(B) ? "" : " after canonicalisation"))

    perm = map(n -> NamedDims.dim_noerror(namesB, n), target)

    if perm == ntuple(identity, ndims(B))
        return B

    elseif !NamedDims.tuple_issubset(namesB, target)
        # Some dimensions of B aren't in the target, add them to the end
        Base.sym_in(:_, target) && error("wildcard in target $target means the result is ambigous")

        extras = filter(n -> n ∉ target, namesB)
        return align(B, (target..., extras...))

    elseif (perm == (2,1) || perm == (0,1)) && eltype(A) <: Number
        return transpose(B)

    elseif 0 in perm
        # Some names in target were not found in B, insert trivial dims
        # C = Transmute{perm}(nameless(B))
        # return NamedDimsArray{target}(C)

        # Fancier version which trims trailing dims, and may return transpose etc.
        short_perm = _trim_trailing_zeros(perm)
        L = ntuple(length(short_perm)) do i
            n = target[i]
            Base.sym_in(n, namesB) ? n : :_
        end
        if length(short_perm) < length(perm)
            return align(B, L)
        else
            C = Transmute{short_perm}(nameless(B))
            return NamedDimsArray{L}(C)
        end

    else
        C = PermutedDimsArray(nameless(B), perm)
        return NamedDimsArray{target}(C)
    end
end

_trim_trailing_zeros(tup) = reverse(_trim_leading_zeros(reverse(tup)...))
_trim_leading_zeros(x, ys...) = x==0 ? _trim_leading_zeros(ys...) : (x, ys...)
_trim_leading_zeros() = ()
# @btime _trim_trailing_zeros((1,2,0,3,0)) # 1μs

# https://github.com/JuliaLang/julia/pull/32968
Base.filter(f, xs::Tuple) = Base.afoldl((ys, x) -> f(x) ? (ys..., x) : ys, (), xs...)
Base.filter(f, t::Base.Any16) = Tuple(filter(f, collect(t)))


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

#################### REDUCE ####################

"""
    sum!ᵃ(A, B)

This variant of `sum!` automatically `align`s dimensions by name.
See also `prod!ᵃ`, and `*ᵃ`.
"""
sum!ᵃ(A::NamedDimsArray, B::NamedDimsArray) = sum!(A, align(B, getnames(A)))
prod!ᵃ(A::NamedDimsArray, B::NamedDimsArray) = prod!(A, align(B, getnames(A)))

#################### CANONICALISE ####################

"""
    A′ = canonise(A::NamedDimsArray)

Re-arranges the index names of `A` to canonical order,
meaning that they now match the underlying storage,
removing lazy re-orderings such as `Transpose`.

This should not affect any operations which work on the index names,
but will confuse anything working on index positions.

`NamedDimsArray{L,T,2,Diagonal}` is unwrapped only if it has two equal names,
which `Diagonal{T,NamedVec}` always has.
"""
canonise(x) = begin
    # @info "nothing to canonicalise?" typeof(x)
    x
end

grandparent(x) = parent(parent(x))

# diagonal / Diagonal
canonise(x::Diagonal{T,<:NamedDimsArray}) where {T} = parent(x)

canonise(x::NamedDimsArray{L,T,2,<:Diagonal}) where {L,T} =
    L[1] === L[2] ? NamedDimsArray{(L[1],)}(grandparent(x)) : x

# transpose / Transpose of a matrix (of numbers, to avoid recursion)
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,2}}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(grandparent(x))

canonise(x::Transpose{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(grandparent(x))

# transpose / Transpose of a vector: drop :_ dimension
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,1}}}) where {L,T<:Number} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(grandparent(x)) : x

canonise(x::Transpose{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Number} = parent(x)

# same for Adjoint but only on reals
canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,2}}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(xgrandparent(x))

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(grandparent(x))

canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,1}}}) where {L,T<:Real} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(grandparent(x)) : x

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Real} = parent(x)

# *mutedDimsArray
canonise(x::PermutedDimsArray{T,N,P,Q,<:NamedDimsArray{L}}) where {L,T,N,P,Q} =
    NamedDimsArray{L}(grandparent(x))

canonise(x::NamedDimsArray{L,T,N,<:PermutedDimsArray{T,N,P,Q}}) where {L,T,N,P,Q} =
    NamedDimsArray{ntuple(i->L[Q[i]],N)}(grandparent(x))

canonise(x::TransmutedDimsArray{T,N,P,Q,<:NamedDimsArray{L}}) where {L,T,N,P,Q} =
    NamedDimsArray{L}(grandparent(x))

canonise(x::NamedDimsArray{L,T,N,<:TransmutedDimsArray{T,N,P,Q}}) where {L,T,N,P,Q} =
    NamedDimsArray{ntuple(i->L[Q[i]], ndims(grandparent(x)))}(grandparent(x))


####################
