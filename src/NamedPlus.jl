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

include("permute.jl")

include("create.jl")

include("reshape.jl")

include("rename.jl")

# include("plots.jl")

end # module
