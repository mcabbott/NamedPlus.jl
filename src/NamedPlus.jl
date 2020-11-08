module NamedPlus

using NamedDims
export NamedDims, NamedDimsArray, rename

# using AxisKeys # perhaps they can co-operate for now
# export wrapdims, KeyedArray, axiskeys

include("recursion.jl")
export getnames, nameless

include("int.jl")
export NamedInt, ᵅ

include("macro.jl")
export @named

include("permute.jl")
export canonise, align, align_sum!, align_prod!

include("create.jl")
export named, diagonal, outer, .., @pirate

include("reshape.jl")

include("rename.jl")
export prime

include("mul.jl")
export mul, *ᵃ

include("show.jl")

using Requires

export contract, ⊙ᵃ, batchmul

@init @require TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
    include("contract.jl")
end

@init @require OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922" begin
    include("omeinsum.jl")
end

@init @require AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5" begin
    include("plots.jl")
end

end # module
