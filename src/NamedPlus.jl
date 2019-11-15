module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, dim, unname, rename

# using AxisRanges # perhaps they can co-operate for now
# export wrapdims, RangeArray, ranges

include("recursion.jl")
export getnames, nameless

include("int.jl")
export NamedInt

include("macro.jl")
export @named

include("permute.jl")
export canonise, align, sum!ᵃ, prod!ᵃ

include("create.jl")
export named, diagonal, outer

include("reshape.jl")

include("rename.jl")
export prime

include("mul.jl")
export mul, *ᵃ

# include("plots.jl")

include("show.jl")

end # module
