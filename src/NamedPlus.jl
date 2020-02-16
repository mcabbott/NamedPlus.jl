module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, rename

# using AxisRanges # perhaps they can co-operate for now
# export wrapdims, RangeArray, ranges

include("recursion.jl")
export getnames, nameless

include("int.jl")
export NamedInt

include("macro.jl")
export @named

include("permute.jl")
export canonise, align, align_sum!, align_prod!

include("create.jl")
export named, diagonal, outer

include("reshape.jl")

include("rename.jl")
export prime

include("mul.jl")
export mul, *ᵃ

include("show.jl")

using Requires

function contract end
export contract, batchmul

@init @require TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
    include("contract.jl")
end

@init @require OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922" begin
    include("omeinsum.jl")
end

@init @require AxisRanges = "7d985058-612f-5500-9f06-de9955ae0899" begin
    include("plots.jl")
end

end # module
