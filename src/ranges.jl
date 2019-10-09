#################### RANGEWRAP ####################

export ranges, getranges, hasranges, rangeless, Wrap, RangeWrap

using OffsetArrays

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
    @boundscheck checkbounds(A.data, I...)
    data = @inbounds getindex(A.data, I...)
    @boundscheck checkbounds.(A.ranges, I)
    ranges = @inbounds range_getindex(A.ranges, I)
    ranges isa Tuple{} ? data : RangeWrap(data, ranges, A.meta)
end
@inline function Base.view(A::RangeWrap, I...)
    @boundscheck checkbounds(A.data, I...)
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
    hasnames(A) || error("must have names!")
    ds = NamedDims.dim(getnames(A), keys(kw))
    inds = ntuple(d -> d in ds ? values(kw)[d] : (:), ndims(A))
    Base.getindex(A, inds...)
end

range_getindex(ranges, inds) = filter(r -> r isa AbstractArray, getindex.(ranges, inds))
range_view(ranges, inds) = filter(r -> ndims(r)>0, view.(ranges, inds))

#=
Wrap(rand(2,2), a=nothing, b=nothing)[:, [CartesianIndex()], :]

Fails at checkbounds.(A.ranges, I),
and then at range_getindex
Need @generated ranges_checkbounds & range_getindex?
=#

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
        if axes(A,d) != axes(r,1) # error("range's axis does not match array's")
            r = OffsetArray(r, axes(A,d))
        end
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
hasranges(x::RangeWrap) = true
hasranges(x::AbstractArray) = x === parent(x) ? false : hasranges(parent(x))

@doc ranges_doc
getranges(x::AbstractArray) = x === parent(x) ? default_ranges(x) :
    outmap(x, getranges(parent(x)), Base.OneTo(1))
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
getmeta(x::AbstractArray) = x === parent(x) ? nothing : getmeta(parent(x))
getmeta(x::RangeWrap) = x.meta

@doc meta_doc
addmeta(x::RangeWrap, meta) = RangeWrap(x.data, x.ranges, meta)
addmeta(x::AbstractArray, meta) = x === parent(parent(x)) ? RangeWrap(x, axes(x), meta) :
    rewraplike(x, parent(x), addmeta(parent(x), meta))

"""
    rangeless(A)

Like nameless but for ranges.
"""
rangeless(x::RangeWrap) = parent(x)
rangeless(x) = x
function rangeless(x::AbstractArray)
    hasranges(x) || return x
    x === parent(x) ? x : rewraplike(x, parent(x), rangeless(parent(x)))
end

rewraplike(x::RangeWrap, y, z) = RangeWrap(z, x.ranges, x.meta)

#################### ROUND BRACKETS ####################

if VERSION >= v"1.3.0-rc2.0"
    Base.@propagate_inbounds (A::RangeUnion)(args...) = get_from_args(A, args...)
    Base.@propagate_inbounds (A::RangeUnion)(;kw...) = get_from_kw(A, kw)
end

"""
    (A::RangeUnion)("a", 2.0, :γ) == A[1, 2, 3]
    A(:γ) == view(A, :,:,3)

`RangeWrap` arrays are callable, and this behaves much like indexing,
except using the contents of the ranges, not the integer indices.

When all `ranges(A)` have distinct `eltype`s,
then a single index may be used to indicate a slice.
"""
Base.@propagate_inbounds (A::RangeWrap)(args...) = get_from_args(A, args...)

Base.@propagate_inbounds function get_from_args(A, args...)
    ranges = getranges(A)

    if length(args) == ndims(A)
        inds = map((v,r) -> findindex(v,r), args, ranges)
        # any(inds .=== nothing) && error("no matching entries found!") # very slow!
        # @boundscheck checkbounds(A, inds...) # TODO add methods to checkbounds for nothing?
        # return @inbounds getindex(A, inds...)
        return getindex(A, inds...)


    elseif length(args)==1 && allunique_types(map(eltype, ranges)...)
        d = findfirst(T -> args[1] isa T, eltype.(ranges))
        i = findindex(first(args), ranges[d])
        inds = ntuple(n -> n==d ? i : (:), ndims(A))
        # @boundscheck checkbounds(A, inds...)
        # return @inbounds getindex(A, inds...)
        return getindex(A, inds...)

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
Base.@propagate_inbounds (A::RangeWrap)(;kw...) = get_from_kw(A, kw)

Base.@propagate_inbounds function get_from_kw(A, kw)
    hasnames(A) || error("named indexing requires a named object!")
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

This is usually `findfirst(isequal(key), range)`,
but understands `findindex(:, range)`
and `findindex(array, range) = intersect(...)`.

If passed a function, `findindex(<(4), range) = findall(x -> x<4, range)`.
Selectors like `All(key)` and `Between(lo,hi)` also call `findall`.
"""
findindex(a, r::AbstractArray) = findfirst(isequal(a), r)

findindex(a::Colon, r::AbstractArray) = Colon()

findindex(a::AbstractArray, r::AbstractArray) = intersect(a, r)

findindex(f::Function, r::AbstractArray) = findall(f, r)


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

for equal in (isequal, Base.:(==))
    @eval begin

# findfirst returns always Int or Nothing

        findfirst(eq::Base.Fix2{typeof($equal),Int}, r::Base.OneTo{Int}) =
            1 <= eq.x <= r.stop ? eq.x : nothing

        findfirst(eq::Base.Fix2{typeof($equal)}, r::AbstractUnitRange) =
            first(r) <= eq.x <= last(r) ? 1+Int(eq.x - first(r)) : nothing

# findall returns a vector... which I would like to make a range?

        findall(eq::Base.Fix2{typeof($equal),Int}, r::Base.OneTo{Int}) =
            1 <= eq.x <= r.stop ? (eq.x:eq.x) : (1:0) # 0.03ns
            # 1 <= eq.x <= r.stop ? [eq.x] : Int[]    # 26 ns

        function findall(eq::Base.Fix2{typeof($equal)}, r::AbstractUnitRange)
            val = 1 + Int(eq.x - first(r))
            first(r) <= eq.x <= last(r) ? (val:val) : (1:0)
            # first(r) <= eq.x <= last(r) ? [val] : Int[]
        end

    end
end

findall(eq::Base.Fix2{typeof(<=),Int}, r::Base.OneTo{Int}) =
    eq.x < 1 ? Base.OneTo(0) :
    eq.x >= r.stop ? r :
    Base.OneTo(eq.x)

findall(eq::Base.Fix2{typeof(<=)}, r::UnitRange{T}) where T =
    intersect(Base.OneTo(trunc(T,eq.x) - first(r) + 1), eachindex(r))

findall(eq::Base.Fix2{typeof(<),Int}, r::Base.OneTo{Int}) =
    eq.x <= 1 ? Base.OneTo(0) :
    eq.x > r.stop ? r :
    Base.OneTo(eq.x - 1)

findall(eq::Base.Fix2{typeof(<)}, r::UnitRange{T}) where T =
    intersect(Base.OneTo(trunc(T,eq.x - first(r))), eachindex(r))

findall(eq::Base.Fix2{typeof(>=),Int}, r::Base.OneTo{Int}) =
    eq.x <= 1 ? UnitRange(r) :
    eq.x > r.stop ? (1:0) :
    (eq.x : r.stop)

findall(eq::Base.Fix2{typeof(>=)}, r::UnitRange{T}) where T =
    intersect(UnitRange(trunc(T,eq.x) - first(r) + 1, length(r)), eachindex(r))

# for more in (is, <)
#     @eval begin

findall(eq::Base.Fix2{typeof(>),Int}, r::Base.OneTo{Int}) =
    eq.x < 1 ? UnitRange(r) :
    eq.x >= r.stop ? (1:0) :
    (eq.x+1 : r.stop)

findall(eq::Base.Fix2{typeof(>)}, r::OrdinalRange{T}) where T =
    intersect(UnitRange(trunc(T,eq.x) - first(r) + 2, length(r)), eachindex(r))


#=
import Base: findfirst, findall, OneTo

@btime findfirst(isequal(300), OneTo(1000)) #  173.225 ns -> 0.029 ns
@btime findfirst(isequal(300), 0:1000)      #   90.410 ns -> 0.029 ns

@btime findall( <(10), OneTo(1000)); # 742.162 ns  ->  0.029 ns
@btime collect(findall( <(10), OneTo(1000))); #    -> 28.855 ns

@btime findall( <(10), 1:1000);   # 890.300 ns -> 0.029 ns
@btime collect(findall( <(10), 1:1000));    # -> 28.888 ns

=#


# https://github.com/JuliaLang/julia/pull/32968
filter(args...) = Base.filter(args...)
filter(f, xs::Tuple) = Base.afoldl((ys, x) -> f(x) ? (ys..., x) : ys, (), xs...)
filter(f, t::Base.Any16) = Tuple(filter(f, collect(t)))

mod(args...) = Base.mod(args...)
mod(i::Integer, r::Base.OneTo) = mod1(i, last(r))
mod(i::Integer, r::AbstractUnitRange{<:Integer}) = mod(i-first(r), length(r)) + first(r)

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
    if hasranges(B)
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

#################### FUNCTIONS ####################

# function Base.mapreduce(f, op, A::RangeWrap; dims=:)
#     dims === Colon() && return mapreduce(f, op, A.data)

#     numerical_dims = hasnames(A) ? NamedDims.dim(getnames(A), dims) : dims
#     data = mapreduce(f, op, A.data; dims=numerical_dims)
#     ranges = ntuple(d -> d in numerical_dims ? Base.OneTo(1) : A.ranges[d], ndims(A))
#     RangeWrap(data, ranges, A.meta)
# end

function Base.mapreduce(f, op, A::PlusUnion; dims=:) # sum, prod, etc
    B = nameless(rangeless(A))
    dims === Colon() && return mapreduce(f, op, B)

    numerical_dims = hasnames(A) ? NamedDims.dim(getnames(A), dims) : dims
    C = mapreduce(f, op, B; dims=numerical_dims)

    X = hasnames(A) ? NamedDimsArray(C, getnames(A)) : C
    if hasranges(A)
        ranges = ntuple(d -> d in numerical_dims ? Base.OneTo(1) : getranges(A)[d], ndims(A))
        return RangeWrap(X, ranges, getmeta(A))
    else
        return X
    end
end

function Base.dropdims(A::RangeWrap; dims)
    numerical_dims = hasnames(A) ? NamedDims.dim(getnames(A), dims) : dims
    data = dropdims(A.data; dims=dims)
    ranges = range_skip(A.ranges, numerical_dims...)
    RangeWrap(data, ranges, A.meta)
end
range_skip(tup::Tuple, d, dims...) = range_skip(
    ntuple(n -> n<d ? tup[n] : tup[n+1], length(tup)-1),
    map(n -> n<d ? n : n-1, dims)...)
range_skip(tup::Tuple) = tup

function Base.permutedims(A::RangeWrap, perm)
    numerical_perm = hasnames(A) ? NamedDims.dim(getnames(A), perm) : perm
    data = permutedims(A.data, numerical_perm)
    ranges = ntuple(d -> A.ranges[findfirst(isequal(d), perm)], ndims(A))
    RangeWrap(data, ranges, A.meta)
end

for fun in (:(Base.permutedims), :(LinearAlgebra.transpose))
    @eval function $fun(A::RangeWrap)
        data = $fun(A.data)
        ranges = ndims(A)==1 ? (Base.OneTo(1), A.ranges[1]) : reverse(A.ranges)
        RangeWrap(data, ranges, A.meta)
    end
end

#################### BROADCASTING ####################
# https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array-1

#=
Base.BroadcastStyle(::Type{<:RangeWrap}) = Broadcast.ArrayStyle{RangeWrap}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{RangeWrap}}, ::Type{ElType}) where ElType
    A = find_aac(bc)
    RangeWrap(similar(Array{ElType}, axes(bc)), A.ranges, A.meta)
end

"`A = find_aac(As)` returns the first RangeWrap among the arguments."
find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(a::RangeWrap, rest) = a
find_aac(::Any, rest) = find_aac(rest)
=#

#################### THE END ####################

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

