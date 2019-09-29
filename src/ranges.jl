# This isn't yet included, needs to be pasted into the REPL

using NamedPlus, LinearAlgebra
using TransmuteDims # https://github.com/mcabbott/TransmuteDims.jl
using NamedPlus: NamedUnion, hasnames, outmap, True

export ranges, Wrap

struct RangeWrap{T,N,AT,RT} <: AbstractArray{T,N}
    data::AT
    ranges::RT
end

function RangeWrap(x::AbstractArray{T,N}, r::Tuple) where {T,N}
    @assert length(r) == N
    RangeWrap{T,N,typeof(x), typeof(r)}(x, r)
end

Base.size(x::RangeWrap) = size(x.data)
Base.axes(x::RangeWrap) = axes(x.data)
Base.parent(x::RangeWrap) = x.data
Base.IndexStyle(A::RangeWrap) = IndexStyle(A.data)

@inline function Base.getindex(A::RangeWrap, I...)
    @boundscheck checkbounds(A, I...)
    data = @inbounds getindex(A.data, I...)
    @boundscheck checkbounds.(A.ranges, I)
    ranges = @inbounds range_getindex(A.ranges, I)
    ranges isa Tuple{} ? data : RangeWrap(data, ranges)
end
@inline function Base.view(A::RangeWrap, I...)
    @boundscheck checkbounds(A, I...)
    data = @inbounds view(A.data, I...)
    @boundscheck checkbounds.(A.ranges, I)
    ranges = @inbounds range_view(A.ranges, I)
    ranges isa Tuple{} ? data : RangeWrap(data, ranges)
end
@inline function Base.setindex!(A::RangeWrap, val, I...)
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.data, val, I...)
    val
end

ranges(x::RangeWrap) = x.ranges
ranges(x, d::Int) = d <= ndims(x) ? ranges(x)[d] : Base.OneTo(1)

range_getindex(ranges, inds) = TransmuteDims.filter(r -> r isa AbstractArray, getindex.(ranges, inds))
range_view(ranges, inds) = TransmuteDims.filter(r -> ndims(r)>0, view.(ranges, inds))


"""
    (A::RangeWrap)("a", 2.0, :γ) == A[1:1, 2:2, 3:3]
    A(:γ) == view(A, :,:,3:3)

`RangeWrap` arrays are callable, and this behaves much like indexing,
except using the contents of the ranges, not the integer indices.
When all `ranges(A)` have distinct `eltype`s,
then a single index indicates a slice.

TODO this should work on RangeUnion?
TODO Base.@propagate_inbounds? @boundscheck?
"""
function (A::RangeWrap)(args...)
    if length(args) == ndims(A)
        inds = Tuple(findall(isequal(a), r) for (a,r) in zip(args, A.ranges))
        return getindex(A.data, inds...)

    elseif length(args)==1 && alldifferent(eltype.(A.ranges)...)
        d = findfirst(T -> args[1] isa T, eltype.(A.ranges))
        i = findall(isequal(first(args)), A.ranges[d])
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

alldifferent(args...) = sum(x===y for x in args, y in args) == length(args)

"""
    findindex(key, range)

This is `findfirst` when `range::AbstractRange` or `key::Symbol`,
otherwise it is `findall`.

TODO allow things like 'a':'z' perhaps? [:a, :b]?
"""
findindex(a, r::AbstractRange) = findfirst(isequal(a), r)
findindex(a::Symbol, r::AbstractArray) = findfirst(isequal(a), r)

findindex(a::Colon, r::AbstractArray) = Colon()

findindex(a, r) = findall(isequal(a), r)

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

function check_names(A, names)
    ndims(A) == length(names) || error("wrong number of names")
    allunique(names) || @warn "not sure how well repeated names will work here"
    names
end

function check_ranges(A, ranges)
    ndims(A) == length(ranges) || error("wrong number of ranges")
    size(A) == length.(ranges) || error("wrong length of ranges")
    for r in ranges
        if eltype(r) == Symbol
            allunique(r) || error("ranges of Symbols need to be unique")
        end
    end
    ranges
end


"""
    hasranges(A)

Returns `True()` if `A::RangeWrap`
or any wrapper around that which defines `parent`,
otherwise `false`.
"""
hasranges(x::RangeWrap) = True()

hasranges(x::AbstractArray) = x === parent(x) ? false : hasnames(parent(x))

"""
    ranges
"""
function ranges(x::AbstractArray)
    # hasranges(x) === True() || return default_ranges(x)
    p = parent(x)
    x === parent(x) && return default_ranges(x)
    return outmap(x, ranges(p), Base.OneTo(1))
end

default_ranges(x::AbstractArray) = axes(x)

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
    return rewraplike(x, p, rangeless(p))
end

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

if VERSION >= v"1.3"
    """
        (A::RangeUnion)(j='b') == A(:, 'b') == A[:, 2:2]

    When you have both names and ranges, you can call `A` with keyword args.
    """
    (A::RangeUnion)(;kw...) = get_from_kw(A, kw)
else
    (A::RangeWrap)(;kw...) = get_from_kw(A, kw)
end

function get_from_kw(A, kw)
    hasnames(A) === True() || error("named indexing requires a named object!")
    list = thenames(A)
    issubset(kw.itr, list) || error("some keywords not in list of names!")
    args = map(s -> Base.sym_in(s, kw.itr) ? getfield(kw.data, s) : Colon(), list)
    A(args...)
end


#=

A = RangeWrap(rand(3), (10:10:30,))
A(10) == A[1:1] # would prefer scalar

B = RangeWrap(rand(2,3), (["a", "b"], 10:10:30,))
B("a")
B(20)
B("b", 20) == B[2:2, 2:2]

C = Wrap(rand(1:99,2,3), i=10:11, j=[:a,:b,:c])
C[1,1]
C(i=10, j=:a)

=#

