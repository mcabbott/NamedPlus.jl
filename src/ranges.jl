# This isn't yet included, needs to be pasted into the REPL

using NamedPlus, LinearAlgebra
using TransmuteDims#master
using NamedPlus: NamedUnion, hasnames, outmap, True

#################### RANGEWRAP ####################

export ranges, getranges, hasranges, Wrap

struct RangeWrap{T,N,AT,RT,MT} <: AbstractArray{T,N}
    data::AT
    ranges::RT
    meta::MT
end

function RangeWrap(data::AbstractArray{T,N}, ranges::Tuple, meta=nothing) where {T,N}
    @assert length(ranges) == N
    RangeWrap{T,N,typeof(data), typeof(ranges), typeof(meta)}(data, ranges, meta)
end

Base.size(x::RangeWrap) = size(x.data)
Base.size(x::RangeWrap, d) = size(x.data, d)

Base.axes(x::RangeWrap) = axes(x.data)
Base.axes(x::RangeWrap, d) = axes(x.data, d)

Base.parent(x::RangeWrap) = x.data

Base.IndexStyle(A::RangeWrap) = IndexStyle(A.data)

@inline function Base.getindex(A::RangeWrap, I...)
    @boundscheck checkbounds(A, I...)
    data = @inbounds getindex(A.data, I...)
    @boundscheck checkbounds.(A.ranges, I)
    ranges = @inbounds range_getindex(A.ranges, I)
    ranges isa Tuple{} ? data : RangeWrap(data, ranges, A.meta)
end
@inline function Base.view(A::RangeWrap, I...)
    @boundscheck checkbounds(A, I...)
    data = @inbounds view(A.data, I...)
    @boundscheck checkbounds.(A.ranges, I)
    ranges = @inbounds range_view(A.ranges, I)
    ranges isa Tuple{} ? data : RangeWrap(data, ranges, A.meta)
end
@inline function Base.setindex!(A::RangeWrap, val, I...)
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.data, val, I...)
    val
end

Base.@propagate_inbounds function Base.getindex(A::RangeWrap; kw...) # untested!
    hasnames(A) === True() || error("must have names!")
    ds = NamedDims.dim(getnames(A), keys(kw))
    inds = ntuple(d -> d in ds ? values(kw)[d] : (:), ndims(A))
    Base.getindex(A, inds...)
end

range_getindex(ranges, inds) = TransmuteDims.filter(r -> r isa AbstractArray, getindex.(ranges, inds))
range_view(ranges, inds) = TransmuteDims.filter(r -> ndims(r)>0, view.(ranges, inds))


"""
    Wrap(A, :i, :j)
    Wrap(A, 1:10, ['a', 'b', 'c'])
    Wrap(A, i=1:10, j=['a', 'b', 'c'])

Function for constructing either a `NamedDimsArray`, a `RangeWrap`,
or a nested pair of both.
"""
Wrap(A::AbstractArray, names::Symbol...) = NamedDimsArray(A, names)
Wrap(A::AbstractArray, ranges::AbstractArray...) = RangeWrap(A, check_ranges(A, ranges))
# Wrap(A::AbstractArray; kw...) =
#     NamedDimsArray(RangeWrap(A, values(kw.data)), check_names(A,kw.itr))
Wrap(A::AbstractArray; kw...) =
    RangeWrap(NamedDimsArray(A, check_names(A,kw.itr)), check_ranges(A, values(kw.data)))
Wrap(A::AbstractArray) = error("you must give some names, or ranges. Or perhaps you wanted `addmeta`?")

function check_names(A, names)
    ndims(A) == length(names) || error("wrong number of names")
    # allunique(names) || @warn "not sure how well repeated names will work here" # and it would be an error before this anyway
    names
end

function check_ranges(A, ranges)
    ndims(A) == length(ranges) || error("wrong number of ranges")
    size(A) == length.(ranges) || error("wrong length of ranges")
    for r in ranges
        if eltype(r) == Symbol
            allunique(r...) || error("ranges of Symbols need to be unique")
        end
    end
    ranges
end

#################### RECURSION ####################

ranges_doc = """
    ranges(A::RangeWrap)
    hasranges(A)
    getranges(A)

`ranges` is the accessor function. `hasranges` is looks inside any wrapper.
`getranges` recurses through wrappers, default is `axes(A)`.
"""
@doc ranges_doc
ranges(x::RangeWrap) = x.ranges
ranges(x, d::Int) = d <= ndims(x) ? ranges(x)[d] : Base.OneTo(1)

@doc ranges_doc
hasranges(x::RangeWrap) = True()

hasranges(x::AbstractArray) = x === parent(x) ? false : hasnames(parent(x))

@doc ranges_doc
function getranges(x::AbstractArray)
    # hasranges(x) === True() || return default_ranges(x)
    p = parent(x)
    x === parent(x) && return default_ranges(x)
    return outmap(x, getranges(p), Base.OneTo(1))
end
getranges(x::RangeWrap) = x.ranges
getranges(x, d::Int) = d <= ndims(x) ? getranges(x)[d] : Base.OneTo(1)


default_ranges(x::AbstractArray) = axes(x)

meta_doc = """
    meta(A::RangeWrap)
    getmeta(A)
    addmeta(A, info)

`getmeta` recursively unwraps, and will return `nothing` if it cannot
find a `RangeWrap` object inside.
`addmeta(A, info)` will add one if it can’t find one.
"""

@doc meta_doc
meta(x::RangeWrap) = x.meta

@doc meta_doc
function getmeta(x::AbstractArray)
    # hasranges(x) === True() || return nothing
    p = parent(x)
    x === parent(x) && return nothing
    return getmeta(x)
end
getmeta(x::RangeWrap) = x.ranges

@doc meta_doc
addmeta(x::RangeWrap, meta) = RangeWrap(x.data, x.ranges, meta)
function addmeta(x::AbstractArray, meta)
    # hasranges(x) === True() || return nothing
    p = parent(x)
    x === parent(x) && return RangeWrap(x, axes(x), meta)
    return NamedPlus.rewraplike(x, p, addmeta(p, meta))
end

"""
    rangeless(A)

Like nameless but for ranges.
"""
rangeless(x::RangeWrap) = parent(x)

rangeless(x) = x

function rangeless(x::AbstractArray)
    hasranges(x) === True() || return x
    p = parent(x)
    p === x && return x
    return NamedPlus.rewraplike(x, p, rangeless(p))
end

#################### UNION TYPES ####################

wraps(AT) = [
    :( Diagonal{<:Any,$AT} ),
    :( Transpose{<:Any,$AT} ),
    :( Adjoint{<:Any,$AT} ),
    :( PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,$AT} ),
    :( TransmutedDimsArray{<:Any,<:Any,<:Any,<:Any,$AT} )
]

@eval begin
    const NamedUnion_maybe_later = Union{
        NamedDimsArray,
        RangeWrap{<:Any,<:Any,<:NamedDimsArray},
        $(wraps(:(<:NamedDimsArray))...),
        $(wraps(:(<:RangeWrap{<:Any,<:Any,<:NamedDimsArray}))...),
    }
    const RangeUnion = Union{
        RangeWrap,
        NamedDimsArray{<:Any,<:Any,<:Any,<:RangeWrap},
        $(wraps(:(<:RangeWrap))...),
        $(wraps(:(<:NamedDimsArray{<:Any,<:Any,<:Any,<:RangeWrap}))...)
    }
end

const PlusUnion = Union{NamedUnion, RangeUnion}

#################### ROUND BRACKETS ####################

if VERSION >= v"1.3.0-rc2.0"
    (A::RangeUnion)(args...) = get_from_args(A, args...)
    (A::RangeUnion)(;kw...) = get_from_kw(A, kw)
end

"""
    (A::RangeUnion)("a", 2.0, :γ) == A[1:1, 2:2, 3:3]
    A(:γ) == view(A, :,:,3:3)

`RangeWrap` arrays are callable, and this behaves much like indexing,
except using the contents of the ranges, not the integer indices.

When all `ranges(A)` have distinct `eltype`s,
then a single index may be used to indicate a slice.

TODO Base.@propagate_inbounds? @boundscheck?
"""
(A::RangeWrap)(args...) = get_from_args(A, args...)

function get_from_args(A, args...)
    ranges = getranges(A)

    if length(args) == ndims(A)
        inds = map((v,r) -> findindex(v,r), args, ranges)
        return getindex(A.data, inds...)

    elseif length(args)==1 && allunique_types(map(eltype, ranges)...)
        d = findfirst(T -> args[1] isa T, eltype.(ranges))
        i = findindex(first(args), ranges[d])
        inds = ntuple(n -> n==d ? i : (:), ndims(A))
        return view(A, inds...)

    end

    if length(args)==1
        error("can only use one entry with all distinct types")
    elseif length(args) != ndims(A)
        error("wrong number of ranges")
    else
        error("can't understand what to do with $args")
    end
end

"""
    (A::RangeUnion)(j='b') == A(:, 'b') == A[:, 2:2]

When you have both names and ranges, you can call `A` with keyword args.
"""
(A::RangeWrap)(;kw...) = get_from_kw(A, kw)

function get_from_kw(A, kw)
    hasnames(A) === True() || error("named indexing requires a named object!")
    list = getnames(A)
    issubset(kw.itr, list) || error("some keywords not in list of names!")
    args = map(s -> Base.sym_in(s, kw.itr) ? getfield(kw.data, s) : Colon(), list)
    A(args...)
end

@generated allunique_types(x, y...) = (x in y) ? false : :(allunique_types($(y...)))
allunique_types(x::DataType) = true

allunique(tup::Tuple) = allunique(tup...)
allunique(x::Symbol, y::Symbol...) = Base.sym_in(x, y) ? false : allunique(y...)
allunique(x::Symbol) = true

#= # Turns out there's a Base.allunique, but it's not so tuple-friently:
@btime allunique(:a, :b, :c)
@btime Base.allunique((:a, :b, :c))
=#

# allunique(x, y...) = (x in y) ? false : allunique(y...)
# allunique(x) = true

"""
    findindex(key, range)

This is essentially `findfirst(isequal(key), range)`,
use `All(key)` for findall.

TODO allow things like 'a':'z' perhaps? [:a, :b]?
"""
findindex(a, r::AbstractRange) = findfirst(isequal(a), r)
findindex(a::Symbol, r::AbstractArray) = findfirst(isequal(a), r)

findindex(a::Colon, r::AbstractArray) = Colon()
findindex(a::Colon, r::AbstractRange) = Colon()

findindex(a, r) = findfirst(isequal(a), r)

#################### SELECTORS ####################

# Selectors alla DimensionalData?
# seems a pain to make these pass through [] indexing,
# perhaps better to reverse: A(3.5, Index(4), "x")

export Near, Between, Index

selector_doc = """
    All(val)
    Near(val)
    Between(lo, hi)

These modify indexing according to `ranges(A)`, to match all instead of first
(it may be a mess right now),
`B(time = Near(3))` (nearest entry according to `abs2(t-3)` of named dimension `:time`)
or `C(iter = Between(10,20))` (matches `10 <= n <= 20`).
"""

abstract type Selector{T} end

@doc selector_doc
struct All{T} <: Selector{T}
    val::T
end
@doc selector_doc
struct Near{T} <: Selector{T}
    val::T
end
@doc selector_doc
struct Between{T} <: Selector{T}
    lo::T
    hi::T
end
Between(lo,hi) = Between(promote(lo,hi)...)

Base.show(io::IO, s::All{T}) where {T} =
    print(io, "All(",s.ind,") ::Selector{",T,"}")
Base.show(io::IO, s::Near{T}) where {T} =
    print(io, "Near(",s.val,") ::Selector{",T,"}")
Base.show(io::IO, s::Between{T}) where {T} =
    print(io, "Between(",s.lo,", ",s.hi,") ::Selector{",T,"}")

# for n in 1:10
#     pre = [Symbol(:i_,i) for i in 1:n]
#     @eval begin
#         Base.getindex(A::RangeWrap, $(pre...), I::Selector, post...) =
#             selector_getindex(A, $(pre...), I, post...)

#         Base.view(A::RangeWrap, $(pre...), I::Selector, post...) =
#             selector_view(A, $(pre...), I, post...)

#         Base.setindex!(A::RangeWrap, val, $(pre...), I::Selector, post...) =
#             selector_setindex!(A, val, $(pre...), I, post...)
#     end
# end
# TODO bounds checking... here, above... not like this:
# Base.checkindex(::Bool, ::AbstractUnitRange, ::Selector) = nothing

# selector_getindex(A, inds...) = getindex(A, selectorise(A, inds...)...)
# selector_view(A, inds...) = view(A, selectorise(A, inds...)...)
# selector_setindex!(A, val, inds...) = setindex(A, val, selectorise(A, inds...)...)

# selectorise(A, inds...) =
#     map(zip(ranges(A), inds)) do (r,i)
#         i isa Selector ? findindex(i, r) : i
#     end # TODO this is slow, it returns an array!

# findindex(sel::At, range::AbstractArray) = findfirst(isequal(sel.val), range)
# findindex(sel::At, range::AbstractRange) = findfirst(isequal(sel.val), range)

findindex(sel::Near, range::AbstractArray) = argmin(map(x -> abs2(x-sel.val), range))
findindex(sel::Near, range::AbstractRange) = argmin(map(x -> abs2(x-sel.val), range))

findindex(sel::Between, range::AbstractArray) = findall(x -> sel.lo <= x <= sel.hi, range)
findindex(sel::Between, range::AbstractRange) = findall(x -> sel.lo <= x <= sel.hi, range)

"""
    Index[i]

This exists to let you mix in square-bracket indexing,
like `A(:b, Near(3.14), Index[4:5], "f")`.
"""
struct Index{T} <: Selector{T}
    ind::T
end

Base.show(io::IO, s::Index{T}) where {T} = print(io, "Index(",s.ind, ")")

Base.getindex(::Type{Index}, i) = Index(i)

findindex(sel::Index, range::AbstractArray) = sel.ind
findindex(sel::Index, range::AbstractRange) = sel.ind


#=

A = RangeWrap(rand(3), (10:10:30,))
A(10) == A[1]

B = Wrap(rand(2,3), ["a", "b"], 10:10:30)
B("a")
B(20)
B("b", 20) == B[2:2, 2]

C = Wrap(rand(1:99,4,3), i=10:13, j=[:a,:b,:c])
C(10, :a)
C(i=10)
C(Near(11.5), :b)
C(Between(10.5, 99), :b)
C(13, Index[3]) == C[4,3]

getnames(Transpose(C))
rangeless(Transpose(C))

nameless(Transpose(C)) # rewrapping fails as RangeWrap not defined within NamedPlus. Damn.

C(j=:b) # no longer crashes julia!

Transpose(C)(j=:b) # ERROR: type Transpose has no field data
# But what should it mean?

=#
