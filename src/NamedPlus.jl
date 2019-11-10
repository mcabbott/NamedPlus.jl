module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, dim, unname

using AxisRanges
export wrapdims, RangeArray, ranges

include("wrap.jl")

export NamedUnion

include("recursion.jl")

export getnames, nameless

include("view.jl")

include("macro.jl")

include("maths.jl")

include("plots.jl")

end # module
