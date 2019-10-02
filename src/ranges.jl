#################### RANGEWRAP ####################

export ranges, getranges, hasranges, rangeless, Wrap, RangeWrap

# mutable struct RangeWrap{T,N,AT,RT,MT} <: AbstractArray{T,N}
#     data::AT
#     ranges::RT
#     meta::MT
# end

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
or a nested pair of both. Performs some sanity checks.

When both are present, it makes a `RangeWrap{...,NamedDimsArray{...}}`
so that it can be callable on Julia < 1.3.
"""
Wrap(A::AbstractArray, names::Symbol...) =
    NamedDimsArray(A, names)
Wrap(A::AbstractArray, ranges::Union{AbstractVector,Nothing}...) =
    RangeWrap(A, check_ranges(A, ranges))
# Wrap(A::AbstractArray; kw...) =
#     NamedDimsArray(RangeWrap(A, values(kw.data)), check_names(A,kw.itr))
Wrap(A::AbstractArray; kw...) =
    RangeWrap(NamedDimsArray(A, check_names(A, kw.itr)), check_ranges(A, values(kw.data)))
# Wrap(A::AbstractArray) = error("you must give some names, or ranges. Or perhaps you wanted `addmeta`?")

function check_names(A, names)
    ndims(A) == length(names) || error("wrong number of names")
    # allunique(names) || @warn "not sure how well repeated names will work here" # and it would be an error before this anyway
    names
end

function check_ranges(A, ranges)
    ndims(A) == length(ranges) || error("wrong number of ranges")
    map(enumerate(ranges)) do (d,r)
        r === nothing && return axes(A,d)
        size(A,d) == length(r) || error("wrong length of ranges")
        if eltype(r) == Symbol
            allunique(r...) || error("ranges of Symbols need to be unique")
        end
        r
    end |> Tuple
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

hasranges(x::AbstractArray) = x === parent(x) ? false : hasranges(parent(x))

@doc ranges_doc
function getranges(x::AbstractArray)
    # hasranges(x) === True() || return default_ranges(x)
    p = parent(x)
    x === p && return default_ranges(x)
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
    x === p && return nothing
    return getmeta(p)
end
getmeta(x::RangeWrap) = x.meta

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
    return rewraplike(x, p, rangeless(p))
end

rewraplike(x::RangeWrap, y, z) = RangeWrap(z, x.ranges, x.meta)

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
"""
@inline (A::RangeWrap)(args...) = get_from_args(A, args...)

@inline function get_from_args(A, args...)
    ranges = getranges(A)

    if length(args) == ndims(A)
        inds = map((v,r) -> findindex(v,r), args, ranges)
        # any(inds .=== nothing) && error("no matching entries found!") # very slow!
        @boundscheck checkbounds(A, inds...) # TODO add methods to checkbounds for nothing?
        # if allint(i...)
            return @inbounds getindex(A.data, inds...)
        # else
        #     return @inbounds view(A.data, inds...)
        # end

    elseif length(args)==1 && allunique_types(map(eltype, ranges)...)
        d = findfirst(T -> args[1] isa T, eltype.(ranges))
        i = findindex(first(args), ranges[d])
        inds = ntuple(n -> n==d ? i : (:), ndims(A))
        @boundscheck checkbounds(A, inds...)
        # return @inbounds view(A, inds...)
        return @inbounds getindex(A, inds...)

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
@inline (A::RangeWrap)(;kw...) = get_from_kw(A, kw)

@inline function get_from_kw(A, kw)
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

allint(x,y...) = x isa Int ? allint(y...) : false
allint() = true

"""
    findindex(key, range)

This is essentially `findfirst(isequal(key), range)`,
use `All(key)` for findall.
"""
findindex(a, r::AbstractArray) = findfirst(isequal(a), r)

findindex(a::Colon, r::AbstractArray) = Colon()

findindex(a::AbstractArray, r::AbstractArray) = intersect(a, r)


#################### SELECTORS ####################

# Selectors alla DimensionalData?
# seems a pain to make these pass through [] indexing,
# perhaps better to reverse: A(3.5, Index(4), "x")

export All, Near, Between, Index

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

findindex(sel::All, range::AbstractArray) = findall(isequal(sel.val), range)

findindex(sel::Near, range::AbstractArray) = argmin(map(x -> abs2(x-sel.val), range))

findindex(sel::Between, range::AbstractArray) = findall(x -> sel.lo <= x <= sel.hi, range)

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

#################### NON-PIRACY ####################

findfirst(args...) = Base.findfirst(args...)
findall(args...) = Base.findall(args...)

findfirst(eq::Base.Fix2{typeof(isequal),Int}, r::Base.OneTo{Int}) =
    1 <= eq.x <= r.stop ? eq.x : nothing

findfirst(eq::Base.Fix2{typeof(isequal),T}, r::AbstractUnitRange{S}) where {T,S} =
    first(r) <= eq.x <= last(r) ? 1+Int(eq.x - first(r)) : nothing

findall(eq::Base.Fix2{typeof(isequal),Int}, r::Base.OneTo{Int}) =
    1 <= eq.x <= r.stop ? (eq.x:eq.x) : nothing   # 0.03ns
    # 1 <= eq.x <= r.stop ? [eq.x] : nothing        # 26 ns


#################### MUTATION ####################

function Base.push!(A::RangeWrap, x)
    push!(A.data, x)
    A.ranges = (extend_one!!(A.ranges[1]),)
    A
end

function Base.pop!(A::RangeWrap)
    out = pop!(A.data)
    A.ranges = (shorten_one!!(A.ranges[1]),)
    out
end

function Base.append!(A::RangeWrap, B)
    push!(A.data, rangeless(B))
    if hasranges(B) === True()
        A.ranges = (append!!(A.ranges[1], getranges(B)[1]),)
        # You could add a branch here for vcat(1:3, 4:5), but prob not worth it.
    else
        A.ranges = (extend_by!!(A.ranges[1], length(B)),)
    end
    A
end

# Like BangBang.jl, these should mutate if they can, but always return result:
extend_one!!(r::Base.OneTo) = Base.OneTo(last(r)+1)
extend_one!!(r::StepRange{Int,Int}) = StepRange(r.start, r.step, r.stop + r.step)
extend_one!!(r::Vector{<:Number}) = push!(r, length(r)+1)
extend_one!!(r::AbstractVector) = vcat(r, length(r)+1)

extend_by!!(r::Base.OneTo, n::Int) = Base.OneTo(last(r)+n)
extend_by!!(r::StepRange{Int,Int}, n::Int) = StepRange(r.start, r.step, r.stop + n * r.step)
extend_by!!(r::Vector{<:Number}) = append!(r, length(r)+1 : length(r)+n+1)
extend_by!!(r::AbstractVector) = vcat(r, length(r)+1 : length(r)+n+1)

append!!(r::Vector, s::AbstractVector) = append!(r,s)
append!!(r::AbstractVector, s::AbstractVector) = vcat(r,s)

shorten_one!!(r::Base.OneTo) = Base.OneTo(last(r)-1)
shorten_one!!(r::Vector) = pop!(r)
shorten_one!!(r::AbstractVector) = r[1:end-1]

#################### PRETTY ####################



# end # module

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


function mysum1(x)
    out = 0.0
    for j in axes(x,2)
        for i in axes(x,1)
            out += x[i,j]
        end
    end
    out
end
function mysum2(x)
    out = 0.0
    for j in axes(x,2)
        for i in axes(x,1)
            @inbounds out += x[i,j]
        end
    end
    out
end
function mysum3(x)
    out = 0.0
    for j in axes(x,2)
        for i in axes(x,1)
            out += x(i,j)
        end
    end
    out
end
function mysum4(x)
    out = 0.0
    for j in axes(x,2)
        for i in axes(x,1)
            @inbounds out += x(i,j)
        end
    end
    out
end

M = rand(1000,1000);
W = Wrap(M, axes(M)...);

@btime mysum1($M) # 1.1 ms
@btime mysum2($M) # 1.1 ms

@btime mysum1($W) # 1.1 ms
@btime mysum2($W) # 1.1 ms

@btime mysum3($W) # 1.677 ms
@btime mysum4($W) # 1.677 ms
=#

#=
# After putting this in a module, etc, still get same re-wrapping error:
using .RangeWrappers: Wrap, rangeless, RangeWrap
using NamedPlus: getnames
NamedPlus.rewraplike(x::RangeWrap, y, z) = Main.RangeWrappers.RangeWrap(z, x.ranges) #
nameless(Transpose(C))
=#


#=
# Tests with AcceleratedArrays
# Summary is that it can speed up All() lookups.

ii = sort(rand(1:25, 100));
D = Wrap(ii .+ (1:100) .* im; i=ii)
j = ii[50]
@btime $D($j)          # 481.267 ns           --> 23.970 ns
@btime $D(i = All($j)) # 295.579 ns           --> 252.365 ns
@btime findall(isequal($j), $ii) # 214.571 ns --> 180.571 ns

using AcceleratedArrays
ii_acc = accelerate(ii, SortIndex);
D_acc = Wrap(ii .+ (1:100) .* im; i=ii_acc);
@btime $D_acc($j)          # 484.118 ns       --> 29.265 ns
@btime $D_acc(i = All($j)) # 131.719 ns
@btime findall(isequal($j), $ii_acc) # 56.227 ns

bb = [string(gensym()) for _ = 1:100];
bb_acc = accelerate(bb, UniqueHashIndex);
bb1 = bb[50]
@btime findfirst(isequal($bb1), $bb)     # 326.094 ns
@btime findall(isequal($bb1), $bb)       # 786.616 ns
@btime findfirst(isequal($bb1), $bb_acc) # 327.374 ns
@btime findall(isequal($bb1), $bb_acc)   #  21.499 ns
BB = Wrap(1:100, bb);
BB_acc = Wrap(1:100, bb_acc);
@btime $BB(All($bb1))         # 824.897 ns
@btime $BB_acc(All($bb1))     #  56.804 ns

=#

