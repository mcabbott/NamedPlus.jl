module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, dim, unname # NamedDims's functions, untouched

export NamedUnion, thenames, nameless # this package's functions

using LinearAlgebra, TransmuteDims

#################### CODE ####################

# Defined here just to include in union types
mutable struct RangeWrap{T,N,AT,RT,MT} <: AbstractArray{T,N}
    data::AT
    ranges::RT
    meta::MT
end

include("wrap.jl") # must be first, defn NamedUnion

include("recursion.jl") # getnames, nameless

include("ranges.jl") # all things RangeWrap

include("view.jl") # permutenames, split/join

include("macro.jl")

include("maths.jl") # contract, svd

#################### BASE.SHOW ####################

function Base.summary(io::IO, x::PlusUnion)
    if hasnames(x) === True()
        if ndims(x)==1
            print(io, length(x),"-element [",summary_pair(getnames(x)[1],axes(x,1)),"] ",typeof(x))
        else
            list = [summary_pair(na,ax) for (na,ax) in zip(getnames(x), axes(x))]
            print(io, join(list," × "), " ",typeof(x))
        end
        names = getnames(x)
    else
        if ndims(x)==1
            print(io, length(x),"-element ",typeof(x))
        else
            print(io, join(size(x)," × "), " ",typeof(x))
        end
        names = Tuple(1:ndims(x))
    end

    if hasranges(x) === True()
        ranges = getranges(x)
        println(io, "\nwith ranges:")
        for d in 1:ndims(x)
            println(io, "    ", names[d], " ∈ ", ranges[d])
        end
    end

    if getmeta(x) !== nothing
        println(io, "and meta:")
        println(io, "    ", repr(getmeta(x)))
    end

    if hasranges(x) === True()
        print(io, "and data")
    end
end

summary_pair(name::Symbol, axis) =
    axis===Base.OneTo(1) ? string(name,"=1") :
    first(axis)==1 ? string(name,"≤",length(axis)) :
    string(name,"∈",first(axis),":",maximum(axis))

#################### THE END ####################

mod(args...) = Base.mod(args...)
mod(i::Integer, r::Base.OneTo) = mod1(i, last(r))
mod(i::Integer, r::AbstractUnitRange{<:Integer}) = mod(i-first(r), length(r)) + first(r)

end # module
