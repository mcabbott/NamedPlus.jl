#################### UNWRAPPING ####################
## https://github.com/invenia/NamedDims.jl/issues/65
# function NamedDims.unname(nda::NamedUnion, names::NTuple{N, Symbol}) where {N}
#     perm = dim(nda, names)
#     if perm == ntuple(identity, N)
#         return unname(nda)
#     elseif perm == (2,1)
#         return transpose(unname(nda))
#     else
#         return PermutedDimsArray(unname(nda), perm)
#     end
# end

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

export permutenames

"""
    permutenames(A, names)
    permutenames(A, names, lazy=false)

This is a bit like `permutedims`, but does not copy the data to a new array in the given order,
and instead wraps it in `Transpose` or `PemutedDimsArray` only if nedded.

If `A` has fewer dimensions than `names`, then trivial dimensions are inserted via `GapView`,
which have name `:_`. It's not yet smart enough to leave off trailing trivial dimensions.

This is the default `lazy=true` behaviour. Keyword `lazy=false` will copy only if needed
to avoid these wrappers. This is not exactly `permutedims(A, names)`, as that always copies.

Note that `canonise(A)` unwraps `Diagonal{...,Vector}` and `Transpose{...,Vector}`
to have just one index.
"""
function permutenames(A::NamedUnion{T}, target::NTuple{N,Symbol}; lazy::Bool=true)  where {T,N}
    B = canonise(A)
    # perm = dim(B, names)
    namesB = NamedDims.names(B)
    perm = map(n -> NamedDims.dim_noerror(namesB, n), target)
    # now check if you got them all? or check issubset first?
    if perm == ntuple(identity, ndims(B))
        return B
    elseif perm == (2,1) && T <: Number
        return lazy ? transpose(B) : copy(transpose(B))
    else
        # C = lazy ? PermutedDimsArray(unname(B), perm) : permutedims(unname(B), perm)
        # return NamedDimsArray{names}(C)
        L = map(n -> n in namesB ? n : :_, target)
        out = NamedDimsArray{L}(GapView(unname(B), perm))
        return lazy ? out : copy(out)
    end
end

# Identical except passes other things through
_permutenames(A::NamedUnion, n::NTuple{N,Symbol} where N) = permutenames(A, n)
_permutenames(A, n::NTuple{N,Symbol} where N) = A

#=
@generated function permutenames(A::NamedUnion{T}, ::Val{target}, ::Val{lazy} = Val(true)) where {T, target, lazy}
    # canonise TODO
    B = A

    namesB = NamedDims.names(B)
    # issubset(namesB, target) || error("all of A's nontrivial names must appear in the target list")
    perm = dim(B, target)
    perm_plus = map(n -> NamedDims.dim_noerror(namesB, n), target)

    if perm == ntuple(identity, ndims(B))
        return B

    elseif perm == (2,1) && T <: Number
        then transpose

    elseif no extra dimensions
        then PermutedDimsArray

    elseif perm is sorted && typeof(parent) <: Array?
        then reshape

    else
        GapView
    end
end
=#

# Maybe this should have error checks & setindex!
# And should be in NamedUnion & names & unname etc as another wrapper?
struct GapView{T,N,P,Q,S} <: AbstractArray{T,N}
    parent::S
end

GapView(A::S) where {S<:AbstractArray{T,N}} where {T,N} =
    GapView{T,N,ntuple(identity,N),ntuple(identity,N),S}(A)

GapView(A::S, P::NTuple{N,Int}) where {S<:AbstractArray{T,M}} where {T,N,M} =
    GapView{T,N,zero_nothing(P),Q_from_P(Val(P), Val(M)),S}(A)

zero_nothing(P::Tuple) = map(n -> n===Nothing ? 0 : n, P)

@generated Q_from_P(::Val{P}, ::Val{M}) where {P,M} =
    ntuple(d -> findfirst(isequal(d),P), M)

Base.size(A::GapView{T,N,P}) where {T,N,P} =
    ntuple(d -> P[d]==0 ? 1 : size(A.parent,P[d]), N)

Base.getindex(A::GapView{T,N,P,Q}, iA::Integer...) where {T,N,P,Q} =
    getindex(A.parent, ntuple(d -> iA[Q[d]], ndims(A.parent))...)

Base.getindex(A::GapView, I::CartesianIndex) = getindex(A, Tuple(I)...)

Base.parent(A::GapView) = A.parent

# function GapView(A::NamedDimsArray, inds::NTuple{N,Symbol}) where {N}
#     namesA = NamedDims.names(A)
#     P = map(n -> NamedDims.dim_noerror(namesA, n), inds)
#     L = map(n -> n in namesA ? n : :_, inds)
#     NamedDimsArray{L}(GapView(unname(A), P))
# end
#
# GapView(f, inds::NTuple{N,Symbol}) where {N} = f



#################### SPLIT / COMBINE ####################

"""
    join(A, :i, :j)
    join(A, (:i, :j) => :i⊗j)

This replaces two indices `i,j` with a combined one `i⊗j`, by reshaping etc.
"""
function Base.join(A::NamedUnion, i...)
    error("not yet")
end

"""
    split(A, (:i, :j), (2,3))
    split(A, :i⊗j => (:i, :j), (2,3))

This replaces indiex `i⊗j` with separate `i,j` by reshaping.
The size of the two new indices must be given afterwards; you can write `(2,:)` etc.
"""
function Base.split(A::NamedUnion, i...)
    error("not yet")
end

#################### RENAME ####################

export rename

"""
    rename(A, names) = NamedDims.rename(A, names)

Discards `A`'s dimension names & replaces with the given ones.
Exactly equivalent to `NamedDimsArray(unname(A), names)` I think.

    rename(A, :i => :j)
    A′, B′ = rename(A, B, :i => :j, :j => :k)

Works a bit like `Base.replace` on index names.
If there are several rules, the first matching rule is applied to each index, not all in sequence.
Given several arrays `A, B`, it makes the same replacements for all, returning a tuple.
"""
rename(nda::NamedUnion, names::NTuple{N, Symbol} where N) = NamedDims.rename(nda, names)

function rename(nda::NamedUnion, pairs::Pair...)
    old = names(nda)
    new = map(old) do i
        for p in pairs
            i === p.first && return p.second
        end
        return i
    end
    NamedDimsArray(unname(nda), new)
end

for n=2:10
    args = [:( $(Symbol("nda_",i))::NamedUnion ) for i=1:n ]
    vals = [:( rename($(Symbol("nda_",i)), pairs...) ) for i=1:n ]
    @eval rename($(args...), pairs::Pair...) = ($(vals...),)
end


#################### PRIMES ####################

export prime

"""
    prime(x, d::Int)
    prime(x, first)
    prime(x, last)
    prime(x, i::Symbol) = rename(x, i => prime(i))

Add a unicode prime `′` to either the indicated index name, or to the given symbol.
Acting on symbols, `prime(s) == Symbol(s, '′')` but faster.
"""
prime(s::Symbol)::Symbol = _prime(Val(s))
# @btime (() -> prime(:a))() # shows 0 allocations
@generated function _prime(vals::Val{s}) where {s}
    QuoteNode(Symbol(s, Symbol('′')))
end

# @btime (() -> _prime((:i,:j,:k), Val(1)))() #  6ns, 1 allocation
_prime(tup::NTuple{N,Symbol}, ::Val{n}) where {N,n} =
    ntuple(i -> i==n ? prime(tup[i])::Symbol : tup[i], N)

# @generated function _prime2(tup::NTuple{N,Symbol}, ::Val{n}) where {N,n}
#     out = [ i==n ? :(_prime(Val(getfield(tup, $i)))) : :(getfield(tup, $i)) for i=1:N ]
#     :(($(out...),))
# end


prime(x::NamedUnion, d::Int) = rename(x, _prime(NamedDims.names(x), Val(d)))
prime(x::NamedUnion, ::typeof(first)) = rename(x, _prime(NamedDims.names(x), Val(1)))
prime(x::NamedUnion, ::typeof(last)) = rename(x, _prime(NamedDims.names(x), Val(ndims(x))))

prime(x::NamedUnion, s::Symbol) = rename(x, s => prime(s))

prime(x::NamedVec) = rename(x, (prime(NamedDims.name(x)[1]),))
# prime(x::Diagonal{??}) =
# prime(x::NamdDimsArray{L,T,N,<:Diagonal{}}) = ??

prime(x::AbstractArray, d) = x

"""
    :x' == :x′

`adjoint(::Symbol)` adds unicode prime `′` to the end.
"""
Base.adjoint(s::Symbol) = prime(s)

####################
