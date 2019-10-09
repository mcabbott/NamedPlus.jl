export stack, allcols, allrows, allslices

#################### STACK TYPE ####################

"""
    stack(::Array{<:Array{T,IN},ON}) <: AbstractArray{T,IN+ON}

Creates a very simple lazy `::Stacked` view of underlying array of arrays.
The dimensions of the inner arrays come first:
```
julia> A = stack([1,2,3] .+ j*im for j=2:2:8)
3×4 Stacked{Complex{Int64},2,Array{Array{Complex{Int64},1},1}}:
 1+2im  1+4im  1+6im  1+8im
 2+2im  2+4im  2+6im  2+8im
 3+2im  3+4im  3+6im  3+8im
```
And acting on a `Generator` it has `collect`ed first.
Some `Base` functions give things from inside:
```
julia> eachcol(A) |> typeof
Array{Array{Complex{Int64},1},1}

julia> view(A, :,1)
3-element Array{Complex{Int64},1}:
 1 + 2im
 2 + 2im
 3 + 2im
```
"""
stack(x::AT) where {AT <: AbstractArray{<:AbstractArray{T,IN},ON}} where {T,IN,ON} =
    Stacked{T, IN+ON, AT}(x)

# struct Stacked{T,N,AT} <: AbstractArray{T,N}
#     slices::AT
# end

#=

using NamedPlus
using NamedPlus: outer_ndims, inner_ndims, Stacked, rewraplike

@named v{j} = [1,2,3]

stack([[1,2,3] for i=1:2])
z = stack([v for i=1:2]); # stackoverflow?
stack(@named [[1,2,3] for i=1:2])
s = stack(@named [v for i=1:2]);

inds = (1,1)

using Debugger
@enter getindex(z, 1,1)

rewraplike(z, parent(z), nameless(parent(z)))

=#

Base.size(x::Stacked) = (size(first(x.slices))..., size(x.slices)...)
Base.axes(x::Stacked) = (axes(first(x.slices))..., axes(x.slices)...)
Base.parent(x::Stacked) = x.slices

outer_ndims(x::Stacked) = ndims(x.slices)
inner_ndims(x::Stacked) = ndims(x) - ndims(x.slices)

function Base.getindex(x::Stacked, inds::Int...)
    IN, ON = inner_ndims(x), outer_ndims(x)
    outer = getindex(x.slices, ntuple(d -> inds[d+IN], ON)...)
    getindex(outer, ntuple(d -> inds[d], IN)...)
end

stack(x::Base.Generator) = stack(collect(x))

Base.eachcol(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}) where {T} = x.slices

Base.view(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}, ::Colon, i::Int) where {T} = x.slices[i]

# To allow general order, I think that like PermutedDimsArray you're going to need P & Q
# but now QI/QO, and P with, umm, some special entries? negative?
# Maybe it's easier to just wrap it if you must?

#################### ALLCOLS ETC ####################

using EllipsisNotation

"""
    allcols(A::Vector{<:Array})
    allrows(A::Vector{<:Array})

Sort-of inverse of `eachcol` / `eachrow`, returns a lazy `Stacked` object,
which is a matrix if `first(A)` is a vector.

    allcols(A::Generator)
    allrows(A::Generator)

Efficiently work through generator, and return a full matrix?
Takes the type from `first(A)`, so it ought to be consistent.

If `ndims(first(A)) > 1` then the index of `A` goes last in `allcols`,
and first in `allrows`.
"""
allcols(x::AbstractVector{<:AbstractArray}) = stack(x)
allrows(x::AbstractVector{<:AbstractVector{<:Number}}) = transpose(stack(x))
function allrows(x::AbstractVector{<:AbstractArray})
    data = stack(x)
    PermutedDimsArray(data, reverse(ntuple(identity, ndims(data))))
end

"""
    allslices(A::Vector{<:Array}; dims)

Sort-of inverse of `eachslice`, returns a lazy `Stacked` object,
permuted such that A's index lies along `dims`.

    allslices(::Generator; dims)

Writes a dense array, maybe more useful.
"""
function allslices(x::AbstractVector{<:AbstractArray}; dims::Int)
    data = stack(x)
    N = ndims(data)
    perm = (ntuple(d -> d<dims ? d : d+1, N-1)..., dims)
    PermutedDimsArray(data, perm)
end

#= # Vector of vectors -- a little quicker if you collect

julia> @btime allcols($(collect(eachcol(rand(10,1000))))); # lazy
  5.611 ns (1 allocation: 16 bytes)

julia> @btime collect(allcols($(collect(eachcol(rand(10,1000))))));
  19.984 μs (3 allocations: 78.22 KiB)

julia> @btime reduce(hcat, $(collect(eachcol(rand(10,1000)))));
  26.781 μs (2 allocations: 78.20 KiB)

=#

function allcols(x::Base.Generator)
    Base.haslength(x) || return allcols(collect(x))
    length(x) == 0 && return allcols(collect(x))
    a = first(x)
    B = similar(a, size(a)..., length(x))
    # copyto!(view(B,..,1), a)
    B[..,1] .= a
    for (i,c) in enumerate(Iterators.drop(x,1))
        # copyto!(view(B,..,i+1), c)
        @inbounds B[..,i+1] .= c
    end
    B
end

function allrows(x::Base.Generator)
    Base.haslength(x) || return allrows(collect(x))
    length(x) == 0 && return allrows(collect(x))
    a = first(x)
    B = similar(a, length(x), size(a)...)
    # copyto!(view(B,1,..), a)
    B[1,..] .= a
    for (i,c) in enumerate(Iterators.drop(x,1))
        # copyto!(view(B,i+1,..), c)
        @inbounds B[i+1,..] .= c
    end
    B
end

function allslices(x::Base.Generator; dims::Int)
    Base.haslength(x) || return allcols(collect(x))
    length(x) == 0 && return allslices(collect(x); dims=dims)
    a = first(x)
    B = similar(a, ntuple(d -> d < dims ? size(a,d) : d > dims ? size(a,d-1) : length(x), ndims(a)+1))
    pre = ntuple(_ -> Colon(), dims - 1)
    post = ntuple(_ -> Colon(), ndims(a) + 1 - dims)

             _allslices(x, a, B, pre, post) end
    function _allslices(x, a, B, pre, post)

    B[pre..., 1, post...] .= a
    for (i,c) in enumerate(Iterators.drop(x,1))
        @inbounds B[pre..., i+1, post...] .= c
    end
    B
end

#= # Generators -- twice as quick.

julia> @btime allcols(eachcol($(rand(10,1000))));
  17.697 μs (1012 allocations: 125.42 KiB)

julia> @btime reduce(hcat, eachcol($(rand(10,1000))));
  7.566 ms (3797 allocations: 38.37 MiB)

julia> @btime reduce(hcat, collect(eachcol($(rand(10,1000)))));
  39.492 μs (1005 allocations: 133.06 KiB)

julia> @btime allslices(eachslice($(rand(10,1000)), dims=2), dims=2); # 581.982 μs without barrier
  26.571 μs (2012 allocations: 172.23 KiB)

=#

#################### NAMES ####################

getnames(x::Stacked) = (getnames(first(x.slices))..., getnames(x.slices)...,)
hasnames(x::Stacked) = hasnames(x.slices) || hasnames(first(x.slices))

getranges(x::Stacked) = (getranges(first(x.slices))..., getranges(x.slices)...,)
hasranges(x::Stacked) = hasranges(x.slices) || hasranges(first(x.slices))

function Base.collect(x::Stacked)
    hasnames(x) || return copy(x)
    NamedDimsArray(copy(x), getnames(x))
end

# allcols(x::NamedDimsArray{})

# allcols on vectors, easy



# Can allcols see deep enough inside a generator to work?



####################
