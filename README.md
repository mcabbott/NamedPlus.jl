# NamedPlus.jl

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 

Things which work:
```julia
@named begin
    m{i,j} = rand(Int8, 3,4)           # matrix whose type has (:i,:j)
    g = [n^i for n in 1:20, i in 1:3]  # read generator's variables
end

t = split(g, :n => (j=4, k=5))   # just reshape, new size (4,5,3)
join(t, (:i, :k) => :χ)          # copy if non-adjacent, (4,15)
ones(z=3) .+ rename(m, :i => :z) # rename m to match ones

d,k = size(m); @show d  # NamedInt, which exists for:
z = zeros(d,d')         # ones, fill, etc, plus ranges:
z .= [sqrt(i) for i in 1:d, i′ in 1:d']   # with PR#81
```

Re-ordering of dimensions:
```julia
permutenames(m, (:j, :k, :i)) # lazy generalised permutedims
@named q{i,j,k} = m .+ t      # use for auto-broadcasting

using TensorOperations
@named @tensor p[j,i′] := m[i,j] * z[i,i′]
```

Some other bits have moved to [AxisRanges.jl](https://github.com/mcabbott/AxisRanges.jl).

Tests are not yet fixed.
