module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, unname, dim

using LinearAlgebra

#################### CODE ####################

include("wrap.jl") # must be first, defn NamedUnion

include("view.jl") # permutenames, split/join

include("macro.jl")

include("maths.jl") # contract, svd

#################### BASE.NAMES ####################

Base.names(x::NamedUnion) = NamedDims.names(x)

Base.names(x::NamedUnion, d::Int) = NamedDims.names(x, d)

NamedDims.names(x, d::Int) = d <= ndims(x) ? NamedDims.names(x)[d] : :_

# Base.getproperty(x::NamedUnion, s::Symbol) =
#     s===:names ? names(x) :
#     getfield(x, s)

# Base.getproperty(x::NamedDimsArray, s::Symbol) =
#     s===:parent ? parent(x) :
#     getfield(x, s)

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

end # module
