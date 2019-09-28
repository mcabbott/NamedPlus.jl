
export ranges

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
    @inbounds getindex(A.data, I...)
end
@inline function Base.setindex!(A::RangeWrap, val, I...)
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.data, val, I...)
    val
end

ranges(x::RangeWrap) = x.ranges
ranges(x, d::Int) = d <= ndims(x) ? ranges(x)[d] : Base.OneTo(1)

# pretty printing, and callable-ness, are the things which want to act on RangeUnion
# also plot recipies, if you go there.

# Base.summary(io::IO, x::RangeWrap) = print(io, "RangeWrap{", typeof(x.data), "}")

function (A::RangeWrap)(args...)
    if length(args) == ndims(A) # then they are positional
        # inds = map((a,r) -> findfirst(isequal(a), r), zip(args, A.ranges))
        inds = Tuple(findfirst(isequal(a), r) for (a,r) in zip(args, A.ranges))
        getindex(A.data, inds...)
    elseif length(args)==1 # && alldifferent(eltype.(A.ranges))
        d = findfirst(T -> args[1] isa T, eltype.(A.ranges))
        i = findfirst(isequal(first(args)), A.ranges[d])
        inds = ntuple(n -> n==d ? i : (:), ndims(A))
        view(A.data, inds...)
    else
        error()
    end
end

struct HasRanges end

"""
    hasranges(A)

Returns `HasRanges()` if `A::RangeWrap`
or any wrapper around that which defines `parent`,
otherwise `false`.
"""
hasranges(x::RangeWrap) = HasRanges()

hasranges(x::AbstractArray) = x === parent(x) ? false : hasnames(parent(x))

"""
    ranges
"""
function ranges(x::AbstractArray)
    # hasranges(x) === HasRanges() || return default_ranges(x)
    p = parent(x)
    x === parent(x) && return default_ranges(x)
    return outmap(x, ranges(p), Base.OneTo(1))
end

default_ranges(x::AbstractArray) = axes(x)

"""
    rangeless(A)
Like nameless
"""
rangeless(x::RangeWrap) = parent(x)

rangeless(x) = x

function rangeless(x::AbstractArray)
    hasranges(x) === HasRanges() || return x
    p = parent(x)
    p === x && return x
    return rewraplike(x, p, rangeless(p))
end

# A = RangeWrap(rand(3), (10:10:30,))
# A(10)
# B = RangeWrap(rand(2,3), (["a", "b"], 10:10:30,))
# B("a")
# B(20)

wraps(AT) = [
    :( Diagonal{<:Any,$AT} ),
    :( Transpose{<:Any,$AT} ),
    :( Adjoint{<:Any,$AT} ),
    :( PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,$AT} ),
    :( TransmutedDimsArray{<:Any,<:Any,<:Any,<:Any,$AT} )
]

@eval begin
    const NamedUnion = Union{
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
