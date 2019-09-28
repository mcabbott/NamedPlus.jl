module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, dim, unname # NamedDims's functions, untouched

export NamedUnion, thenames, nameless # this package's functions

using LinearAlgebra, TransmuteDims

#################### CODE ####################

include("wrap.jl") # must be first, defn NamedUnion

include("recursion.jl") # thenames, nameless

include("view.jl") # permutenames, split/join

include("macro.jl")

include("maths.jl") # contract, svd

#################### BASE.NAMES ####################

Base.names(x::NamedUnion) = thenames(x)

Base.names(x::NamedUnion, d::Int) = thenames(x, d)

thenames(x, d::Int) = d <= ndims(x) ? thenames(x)[d] : :_

#################### BASE.SHOW ####################

function Base.summary(io::IO, x::NamedUnion)
    if ndims(x)==1
        print(io, length(x),"-element (",summary_pair(names(x)[1],axes(x,1)),") ",typeof(x))
    else
        list = [summary_pair(na,ax) for (na,ax) in zip(names(x), axes(x))]
        print(io, join(list," × "), " ",typeof(x))
    end
end

summary_pair(name::Symbol, axis) =
    axis===Base.OneTo(1) ? string(name,"=1") :
    first(axis)==1 ? string(name,"≤",length(axis)) :
    string(name,"∈",first(axis),":",maximum(axis))

#################### THE END ####################

@static if VERSION < v"1.3"
    Base.mod(i::Integer, r::Base.OneTo) = mod1(i, last(r))
    Base.mod(i::Integer, r::AbstractUnitRange{<:Integer}) = mod(i-first(r), length(r)) + first(r)
end

end # module
