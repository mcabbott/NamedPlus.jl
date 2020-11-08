#################### RENAME ####################

"""
    rename(A, names) = NamedDimsArray(nameless(A), names)

Discards `A`'s dimension names & replaces with the given ones.

    rename(A, :i => :j)
    rename(A, :i => :j, :k => :l)
    A′, B′ = rename(A, B, :i => :j)

Works a bit like `Base.replace` on index names.
If there are several rules, they are applied in sequence. (It's fast with up to two!)
Given several arrays `A, B`, it makes the same replacements for all, returning a tuple.
"""
function NamedDims.rename(nda::NamedDimsArray, pair::Pair)
    new = _rename(getnames(nda), pair)
    NamedDimsArray(nameless(nda), new)
end
function NamedDims.rename(nda::NamedDimsArray, pair::Pair, pair2::Pair, rest::Pair...)
    new1 = _rename(getnames(nda), pair)
    new2 = _rename(new1, pair2)
    rename(NamedDimsArray(nameless(nda), new2), rest...)
end
NamedDims.rename(nda::NamedDimsArray) = nda

_rename(old, pair) = map(s -> s===pair.first ? pair.second : s, old) |> NamedDims.compile_time_return_hack
# @btime NamedPlus._rename((:a, :b), :b => :c) # 1.420 ns (0 allocations: 0 bytes)

#=
const ndv = NamedDimsArray{(:a,)}([1,2,3])
@btime (() -> rename(ndv, :a => :b))()            #  6.679 ns (1 allocation: 16 bytes)
@btime (() -> rename(ndv, :a => :b, :b => :c))()  #  6.681 ns (1 allocation: 16 bytes)

=#

for n=2:10
    args = [:( $(Symbol("nda_",i))::NamedDimsArray ) for i=1:n ]
    vals = [:( rename($(Symbol("nda_",i)), pairs...) ) for i=1:n ]
    @eval NamedDims.rename($(args...), pairs::Pair...) = ($(vals...),)
end

#################### PRIMES ####################

"""
    prime(x)

Add a unicode prime `′` to every dimension name.
Acting on symbols, `prime(s) == Symbol(s, '′')` but faster.

    prime(x, d::Int)
    prime(x, first) = prime(x, 1)
    prime(x, last)  = prime(x, ndims(x))
    prime(x, i::Symbol) = rename(x, i => prime(i))

Add a `′` to either the indicated index name, or to the given symbol.
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
    NamedDimsArray(nameless(x), _prime(getnames(x), Val(d)))
prime(x::NamedUnion, ::typeof(first)) =
    NamedDimsArray(nameless(x), _prime(getnames(x), Val(1)))
prime(x::NamedUnion, ::typeof(last)) =
    NamedDimsArray(nameless(x), _prime(getnames(x), Val(ndims(x))))

prime(x::NamedDimsArray{L,T,1}) where {L,T} =
    NamedDimsArray(parent(x), (_prime(Val(L[1])),))

prime(x::NamedUnion, s::Symbol) = rename(x, s => prime(s))

prime(x::AbstractArray, d) = x

prime(x::NamedUnion) = NamedDimsArray(nameless(x), map(prime, getnames(x)))

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


#=
AB = NamedDimsArray(rand(1:10, 2,3), (:a,:b))

@btime prime($AB, first)        # 17 ns, was 1 μs using rename
@code_warntype prime(AB, first) # Body::NamedDimsArray{(:a′, :b),

@btime (AB -> prime(AB, :i))($AB) # 550 ns
=#

####################
