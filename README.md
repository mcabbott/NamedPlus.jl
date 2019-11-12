# NamedPlus.jl

[![Build Status](https://travis-ci.org/mcabbott/NamedPlus.jl.svg?branch=master)](https://travis-ci.org/mcabbott/NamedPlus.jl)

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 

Things which work:
```julia
@named begin
    m{i,j} = rand(Int8, 3,4)             # create a matrix whose type has (:i,:j)
    g = [n^i for n in 1:20, i in 1:3]    # read names from generator's variables
end

t = split(g, :n => (j=4, k=5))           # just reshape, new size (4,5,3)
join(t, (:i, :k) => :χ)                  # copy if non-adjacent, size (4,15)
rename(m, :i => :z)                      # rename just i

d,k = size(m); @show d                   # NamedInt, which exists for:
z = zeros(d,d')                          # ones, fill, etc, plus ranges:
z .= [sqrt(i) for i in 1:d, i′ in 1:d']  # comprehensions have names with PR#81
reshape(g, k,:,d) .+ g[end, d]

ones(i=1, j=4) .+ rand(Int8, i=3)        # base piracy, but convenient.
```

Re-ordering of dimensions:
```julia
align(m, (:j, :k, :i))                   # lazy generalised permutedims
@named q{i,j,k} = m .+ t                 # used for auto-aligned broadcasting
align(m, t) .+ t

using TensorOperations                   # named inputs re-arranged via Strided
@named @tensor p[j,i′] := m[i,j] * z[i,i′]
```

Some other bits have moved to [AxisRanges.jl](https://github.com/mcabbott/AxisRanges.jl).
Which the macro knows about, e.g. `@named [n^i for n in 1:2:20, i in 1:3]` has ranges,
if both packages are loaded.

