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

    if abs(d1 - d2) == 1 && nameless(A) isa StridedArray # is this the right thing?
        sz = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? size(A,d) :
            d==min(d1,d2) ? size(A,d1) * size(A,d2) :
            size(A, d+1)
        end
        nm = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? getnames(A,d) :
            d==min(d1,d2) ? p.second :
            getnames(A, d+1)
        end
        return NamedDimsArray{nm}(reshape(nameless(A), sz))

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
# @btime (() -> _split($(QuoteNode(Symbol("i⊗j")))))()  # 0 allocations, but 4 μs!
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
    split(A, :i => (:i, :_)) == split(A, :i)

The sizes can also bew read from another `B::NamedDimsArray` with indices `i,j`.
If they are omitted, then the second size will be 1.
"""
Base.split(A::NamedUnion, (i,j), sz=nothing) = split(A, _join(i,j) => (i,j), sz)

Base.split(A::NamedUnion, i::Symbol) = split(A, i => (i,:_), nothing)

function Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, sizes::Tuple)
    d0 = dim(A, pair.first)
    sz = ntuple(ndims(A)+1) do d
        d < d0 ? size(A,d) :
        d==d0 ? sizes[1] :
        d==d0+1 ? sizes[2] :
        size(A, d-1)
    end
    nm = ntuple(ndims(A)+1) do d
        d < d0 ? getnames(A,d) :
        d==d0 ? pair.second[1] :
        d==d0+1 ? pair.second[2] :
        getnames(A, d-1)
    end
    NamedDimsArray{nm}(reshape(nameless(A), sz))
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
    rename(A, names) = NamedDimsArray(nameless(A), names)

Discards `A`'s dimension names & replaces with the given ones.
Does this even need its own function?

    rename(A, :i => :j)
    rename(A, :i => :j, :k => :l)
    A′, B′ = rename(A, B, :i => :j)

Works a bit like `Base.replace` on index names. (Could even be made a method of that.)
If there are several rules, they are applied in sequence. (It's fast with up to two!)
Given several arrays `A, B`, it makes the same replacements for all, returning a tuple.
"""
rename(nda::NamedUnion, names::Tuple{Vararg{Symbol}}) = NamedDimsArray(nameless(nda), names)

function rename(nda::NamedUnion, pair::Pair)
    new = _rename(getnames(nda), pair)
    NamedDimsArray(nameless(nda), new)
end
function rename(nda::NamedUnion, pair::Pair, pair2::Pair, rest::Pair...)
    new1 = _rename(getnames(nda), pair)
    new2 = _rename(new1, pair2)
    rename(NamedDimsArray(nameless(nda), new2), rest...)
end
rename(nda::NamedUnion) = nda

_rename(old, pair) = map(s -> s===pair.first ? pair.second : s, old) |> NamedDims.compile_time_return_hack
# @btime NamedPlus._rename((:a, :b), :b => :c) # 1.420 ns (0 allocations: 0 bytes)

#=
const ndv = NamedDimsArray{(:a,)}([1,2,3])
@btime (() -> rename(ndv, :a => :b))()            #  6.679 ns (1 allocation: 16 bytes)
@btime (() -> rename(ndv, :a => :b, :b => :c))()  #  6.681 ns (1 allocation: 16 bytes)

=#

for n=2:10
    args = [:( $(Symbol("nda_",i))::NamedUnion ) for i=1:n ]
    vals = [:( rename($(Symbol("nda_",i)), pairs...) ) for i=1:n ]
    @eval rename($(args...), pairs::Pair...) = ($(vals...),)
end

#################### PRIMES ####################

export prime

"""
    prime(x, d::Int)
    prime(x, first) = prime(x, 1)
    prime(x, last)  = prime(x, ndims(x))
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
    NamedDimsArray{_prime(getnames(x), Val(d))}(nameless(x))
prime(x::NamedUnion, ::typeof(first)) =
    NamedDimsArray{_prime(getnames(x), Val(1))}(nameless(x))
prime(x::NamedUnion, ::typeof(last)) =
    NamedDimsArray{_prime(getnames(x), Val(ndims(x)))}(nameless(x))

prime(x::NamedDimsArray{L,T,1}) where {L,T} =
    NamedDimsArray{(_prime(Val(L[1])),)}(parent(x))

prime(x::NamedUnion, s::Symbol) = rename(x, s => prime(s))

prime(x::AbstractArray, d) = x

#=
# These are zero-cost:
@btime (() -> prime(NamedDimsArray{(:a,)}([1,2,3]), first))()  # 33.237 ns (2 allocations: 128 bytes)
@btime (() -> prime(NamedDimsArray{(:a,)}([1,2,3])))()         # 33.307 ns (2 allocations: 128 bytes)
@btime (() -> NamedDimsArray{(:a,)}([1,2,3]))() # without prime, 33.344 ns (2 allocations: 128 bytes)

# And...
@btime (() -> prime(NamedDimsArray{(:a,)}([1,2,3]), 1))()     # 1.258 μs (9 allocations: 432 bytes)
const ndv = NamedDimsArray{(:a,)}([1,2,3])
@btime (() -> prime(ndv, 1))()                                #  6.678 ns (1 allocation: 16 bytes)
@btime (() -> prime(ndv, :a))()                               #  6.678 ns (1 allocation: 16 bytes)
=#

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
