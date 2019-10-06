#################### NEW EACHSLICE ####################
# https://github.com/JuliaLang/julia/pull/32310/files

export eachcol2, eachrow2, eachslice2

struct EachSlice{A,I,L}
    arr::A # underlying array
    cartiter::I # CartesianIndices iterator
    lookup::L # dimension look up: dimension index in cartiter, or nothing
end

function Base.iterate(s::EachSlice, state...)
    r = iterate(s.cartiter, state...)
    r === nothing && return r
    (c,nextstate) = r
    view(s.arr, map(l -> l === nothing ? (:) : c[l], s.lookup)...), nextstate
end

Base.size(s::EachSlice) = size(s.cartiter)
Base.length(s::EachSlice) = length(s.cartiter)
Base.ndims(s::EachSlice) = ndims(s.cartiter)
Base.IteratorSize(::Type{EachSlice{A,I,L}}) where {A,I,L} = Iterators.IteratorSize(I)
Base.IteratorEltype(::Type{EachSlice{A,I,L}}) where {A,I,L} = Iterators.EltypeUnknown()

Base.parent(s::EachSlice) = s.arr

function eachrow2(A::AbstractVecOrMat)
    iter = CartesianIndices((axes(A,1),))
    lookup = (1,nothing)
    EachSlice(A,iter,lookup)
end
const EachRow{A,I} = EachSlice{A,I,Tuple{Int,Nothing}}
function eachcol2(A::AbstractVecOrMat)
    iter = CartesianIndices((axes(A,2),))
    lookup = (nothing,1)
    EachSlice(A,iter,lookup)
end
const EachCol{A,I} = EachSlice{A,I,Tuple{Nothing,Int}}

@inline function eachslice2(A::AbstractArray; dims)
    for dim in dims
        dim <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    end
    iter = CartesianIndices(map(dim -> axes(A,dim), dims))
    lookup = ntuple(dim -> findfirst(isequal(dim), dims), ndims(A))
    EachSlice(A,iter,lookup)
end

#################### ADDING NAMES ####################

outmap(x::EachSlice, tup) = ntuple(d -> tup[findfirst(isequal(d),x.lookup)], count(!isnothing, x.lookup))
function innernames(x::EachSlice, tup)
    stepone = map((n,t) -> isnothing(n) ? t : nothing, x.lookup, tup)
    NamedPlus.filter(!isnothing, stepone)
end

function Base.collect(itr::Base.Generator{<:NamedDimsArray{L}}) where {L}
    NamedDimsArray{L}(collect(Base.Generator(itr.f, parent(itr.iter))))
end

function Base.collect(itr::Base.Generator{<:EachSlice{<:NamedDimsArray{L}}}) where {L}
    newslices = EachSlice(parent(itr.iter.arr), itr.iter.cartiter, itr.iter.lookup) # nameless?
    innames = innernames(itr.iter, L)
    newfun(x) = itr.f(NamedDimsArray{innames}(x))
    outnames = outmap(itr.iter, L)
    NamedDimsArray{outnames}(collect(Base.Generator(newfun, newslices)))
end

function Base.collect(s::EachSlice{<:NamedDimsArray{L}}) where {L}
    newslices = EachSlice(parent(s.arr), s.cartiter, s.lookup) # should just be nameless?
    innames = innernames(s, L)
    outnames = outmap(s, L)
    NamedDimsArray{outnames}(collect(Base.Generator(NamedDimsArray{innames}, newslices)))
end

#=

ab = NamedDimsArray((1:3) .+ zeros(Int,3)', (:a, :b))

f(x) = (@show typeof(x); sum(x))
[f(x) for x in eachcol(ab)]  # keeps inner
[f(x) for x in eachcol2(ab)] # keeps both now!

=#

####################
