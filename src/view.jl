#################### UNWRAPPING ####################
## https://github.com/invenia/NamedDims.jl/issues/65
# function NamedDims.unname(nda::NamedUnion, names::NTuple{N, Symbol}) where {N}
#     perm = dim(nda, names)
#     if perm == ntuple(identity, N)
#         return unname(nda)
#     elseif perm == (2,1) # note this doesn't catch recursive transpose
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

#=
@inline NamedDims.unname(A::NamedUnion, target::NTuple{N,Symbol} where N) =
    _unname(A, Val(target))

@generated function _unname(A::NamedUnion{T}, vt::Val{target}) where {T, target}
    # canonise TODO
    namesA = NamedDims.names(A)
    perm = _perm(Val(namesA), Val(target))
    :(_unname2(A, Val($perm)))
end
@generated function _unname2(A, ::Val{perm}) where {perm}

    if perm == ntuple(identity, ndims(A))
        return unname(A)

    elseif (perm == (2,1) || perm == (0,1)) && T <: Number
        return :( transpose(unname(A)) )

    elseif !(0 in perm)
        return :( PermutedDimsArray(unname(A), perm) )

    # elseif perm is sorted && typeof(parent) <: Array?
    #     then reshape

    else
        # L = map(n -> n in namesA ? n : :_, targ)
        return :( GapView(unname(A), $perm) )
    end
end

@generated function _perm(::Val{namesA}, ::Val{target}) where {namesA, target}
    fullperm = map(n -> NamedDims.dim_noerror(namesA, n), target)
    perm, targ = chop_zeros(fullperm, target)
    perm
end
=#

#################### PERMUTENAMES ####################

export permutenames

"""
    permutenames(A, names)

This is a bit like `permutedims`, but does not copy the data to a new array in the given order,
and instead wraps it in `Transpose` or `PemutedDimsArray` as nedded.

If `A` has fewer dimensions than `names`, then trivial dimensions are inserted via `GapView`,
which have name `:_`. But trailing dimensions are omitted.

#= This is the default `lazy=true` behaviour. Keyword `lazy=false` will copy only if needed
to avoid these wrappers. This is not exactly `permutedims(A, names)`, as that always copies. =#

Note that `canonise(A)` unwraps `Diagonal{...,Vector}` and `Transpose{...,Vector}`
to have just one index.
"""
function permutenames(A::NamedUnion{T}, target::NTuple{N,Symbol}; lazy::Bool=true)  where {T,N}
    B = canonise(A)

    namesB = NamedDims.names(B)
    fullperm = map(n -> NamedDims.dim_noerror(namesB, n), target)

    perm, targ = chop_zeros(fullperm, target)

    if perm == ntuple(identity, ndims(B))
        return B

    elseif (perm == (2,1) || perm == (0,1)) && T <: Number
        # return lazy ? transpose(B) : permutedims(B)
        return transpose(B)

    elseif 0 in perm
        L = map(n -> n in namesB ? n : :_, targ)
        out = NamedDimsArray{L}(GapView(unname(B), perm))
        # return lazy ? out : copy(out)
        return out

    else
        # C = lazy ? PermutedDimsArray(unname(B), perm) : permutedims(unname(B), perm)
        C = PermutedDimsArray(unname(B), perm)
        return NamedDimsArray{targ}(C)
    end
end

chop_zeros(perm, targ) = last(perm)==0 ? chop_zeros(perm[1:end-1], targ[1:end-1]) : (perm, targ)

# Identical except passes other things through
_permutenames(A::NamedUnion, n::NTuple{N,Symbol} where N) = permutenames(A, n)
_permutenames(A, n::NTuple{N,Symbol} where N) = A

# This is quite slow:
#=
AB = NamedDimsArray(rand(1:10, 2,3), (:a,:b))
@btime (AB -> permutenames(AB, (:b, :z, :a, :c)))($AB) # 6 μs
@btime (AB -> permutenames2(AB, (:b, :z, :a, :c)))($AB) # 9 μs
=#

permutenames2(A::NamedUnion, target::NTuple{N,Symbol} where N) =
    permutenames2(A, Val(target))

@generated function permutenames2(A::NamedUnion{T}, ::Val{target}) where {T, target}
    # canonise TODO

    namesA = NamedDims.names(A)
    fullperm = map(n -> NamedDims.dim_noerror(namesA, n), target)

    perm, targ = chop_zeros(fullperm, target)

    if perm == ntuple(identity, ndims(A))
        return A

    elseif (perm == (2,1) || perm == (0,1)) && T <: Number
        return :( transpose(A) )

    elseif !(0 in perm)
        return :( NamedDimsArray{targ}(PermutedDimsArray(unname(A), perm)) )

    # elseif perm is sorted && typeof(parent) <: Array?
    #     then reshape

    else
        L = map(n -> n in namesA ? n : :_, targ)
        return :( NamedDimsArray{$L}(GapView(unname(A), $perm)) )
    end
end


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
    join(A, (:i, :j) => Symbol("i⊗j"))

This replaces two indices `i,j` with a combined one `i⊗j`, by reshaping.
If the indices aren't adjacent, or `A` is a lazy `Transpose` etc, then it will copy `A`.
"""
Base.join(A::NamedUnion, i::Symbol, j::Symbol) = join(A, (i,j) => _join(i,j))
Base.join(A::NamedUnion, ij::Tuple) = join(A, ij...)

function Base.join(A::NamedUnion, p::Pair{<:Tuple,Symbol})
    d1, d2 = dim(A, p.first)

    if abs(d1 - d2) == 1 && unname(A) isa StridedArray # is this the right thing?
        sz = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? size(A,d) :
            d==min(d1,d2) ? size(A,d1) * size(A,d2) :
            size(A, d+1)
        end
        nm = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? names(A,d) :
            d==min(d1,d2) ? p.second :
            names(A, d+1)
        end
        return NamedDimsArray{nm}(reshape(unname(A), sz))

    elseif abs(d1 - d2) == 1
        return join(copy(A), p)

    else
        s = abs(d1-d2) - 1 # amount to cycle by
        perm = ntuple(ndims(A)) do d
            d < min(d1,d2) ? d :
            d >= max(d1,d2) ? d :
            mod(d-min(d1,d2)+s-1, min(d1,d2):max(d1,d2)-1)
        end
        return join(permutedims(A, perm), p)
    end
    error("not yet")
end

# @btime (() -> _join(:i, :j))() # 0 allocations
_join(i::Symbol, j::Symbol) = _join(Val(i), Val(j))
@generated _join(::Val{i}, ::Val{j}) where {i,j} = QuoteNode(Symbol(i, :⊗, j))

# @btime (() -> _split(Symbol("i⊗j")))()  # 0 allocations, but 4 μs!
# @btime (() -> _split(_join(:i, :j)))()  # 0 allocations, 1.4 ns
_split(ij::Symbol) = _split(Val(ij))
@generated _split(::Val{ij}) where {ij} = Tuple(map(QuoteNode∘Symbol, split(string(ij), '⊗')))

"""
    split(A, (:i, :j), (2,3))
    split(A, Symbol("i⊗j") => (:i, :j), (2,3))

This replaces indiex `i⊗j` with separate `i,j` by reshaping.
The size of the two new indices should be given afterwards;
you may write `(2,:)` etc.

    split(A, (:i, :j), B)
    split(A, :i => (:i, :_))

The sizes can also bew read from another `B::NamedDimsArray` with indices `i,j`.
If they are omitted, then the second size will be 1.
"""
Base.split(A::NamedUnion, (i,j), sz=nothing) = split(A, _join(i,j) => (i,j), sz)

function Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, sizes::Tuple)
    d0 = dim(A, pair.first)
    sz = ntuple(ndims(A)+1) do d
        d < d0 ? size(A,d) :
        d==d0 ? sizes[1] :
        d==d0+1 ? sizes[2] :
        size(A, d-1)
    end
    nm = ntuple(ndims(A)+1) do d
        d < d0 ? names(A,d) :
        d==d0 ? pair.second[1] :
        d==d0+1 ? pair.second[2] :
        names(A, d-1)
    end
    NamedDimsArray{nm}(reshape(unname(A), sz))
end

Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, B::NamedUnion) =
    split(A, pair, dim(B, pair.second))

Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, ::Nothing) =
    split(A, pair, (size(A,pair.first), 1))

#=
ABC = NamedDimsArray(rand(1:10, 2,3,4), (:a,:b,:c))

@btime (ABC -> join(ABC, :a, :b))($ABC)        # 9 μs
@code_warntype (ABC -> join(ABC, :a, :b))(ABC) # ::Any

@btime (ABC -> split(ABC, :c => (:c1, :c2), (2,2)))($ABC)        # 650 ns
@code_warntype (ABC -> split(ABC, :c => (:c1, :c2), (2,2)))(ABC) # ::NamedDimsArray{_A,Int64,_B,_C}
=#

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

prime(x::NamedUnion, d::Int) =
    NamedDimsArray{_prime(NamedDims.names(x), Val(d))}(unname(x))
prime(x::NamedUnion, ::typeof(first)) =
    NamedDimsArray{_prime(NamedDims.names(x), Val(1))}(unname(x))
prime(x::NamedUnion, ::typeof(last)) =
    NamedDimsArray{_prime(NamedDims.names(x), Val(ndims(x)))}(unname(x))

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

#=
AB = NamedDimsArray(rand(1:10, 2,3), (:a,:b))

@btime prime($AB, first)        # 17 ns, was 1 μs using rename
@code_warntype prime(AB, first) # Body::NamedDimsArray{(:a′, :b),

@btime (AB -> prime(AB, :i))($AB) # 550 ns
=#

####################
