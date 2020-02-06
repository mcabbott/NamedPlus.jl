# NamedPlus.jl

[![Build Status](https://travis-ci.org/mcabbott/NamedPlus.jl.svg?branch=master)](https://travis-ci.org/mcabbott/NamedPlus.jl)

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 
Here's what works:

Some convenient ways handle names:
```julia
@named begin
    m{i,j} = rand(Int8, 3,4)             # create a matrix whose type has (:i,:j)
    g = [n^i for n in 1:20, i in 1:3]    # read names from generator's variables
end
ones(i=1, j=4) .+ rand(Int8, i=3)        # base piracy, but convenient.
a_z = named(rand(4,1,1,2), :a, .., :z)   # using EllipsisNotation
dropdims(a_z)                            # defaults to :_, and kills all of them.
transpose(a_z, :a, :z)                   # permutes (4,2,3,1)

t = split(g, :n => (j=4, k=5))           # just reshape, new size (4,5,3)
join(t, (:i, :k) => :χ)                  # copy if non-adjacent, size (4,15)
rename(m, :i => :z)                      # rename just i

d,k = size(m); @show d                   # NamedInt, which exists for:
z = zeros(d,d')                          # ones, fill, etc, plus ranges:
z .= [sqrt(i) for i in 1:d, i′ in 1:d']  # comprehensions propagate names
reshape(g, k,:,d) .+ g[end, d]           # reshape propagate via sizes, as does:
@einsum ζ[i,k] := m[i,j] * z[i,k]        # using Einsum
```

Some automatic re-ordering of dimensions:
```julia
align(m, (:j, :k, :i))                   # lazy generalised permutedims
@named q{i,j,k} = m .+ t                 # used for auto-permuted broadcasting
align(m, t) .+ t                         # or to manually fix things up

sum!ᵃ(Int.(m), t)                        # reduce (:j, :k, :i) into (:i, :j)

m *ᵃ z == mul(m, z, :i) == m' * z        # matrix multiplication on shared index
g *ᵃ m == (m *ᵃ g)'

using TensorOperations                   # named inputs re-arranged via Strided
@named @tensor p[j,i′] := m[i,j] * z[i,i′]
contract(m, t)                           # shared indices i & j, leaving only k
```

Some other bits have moved to [AxisRanges.jl](https://github.com/mcabbott/AxisRanges.jl).
If both packages are loaded:

```julia
using NamedPlus, AxisRanges, Plots
@named [n^i for n in 1:2:40, i in 2:4]   # has custom ranges
scatter(ans, yaxis=:log10)
```

Compared to [Pytorch](https://pytorch.org/docs/stable/named_tensor.html)'s new named tensors: 

* `refine_names` ⟶ `named`, except with `..` instead of `...`.
* `unflatten` ⟶ `split` exactly, and `flatten` ⟶ `join`, except that for them "All of dims must be consecutive in order" while mine permutes if required.
* `.align_to` and `.align_as` ⟶ `align`, mine allows the target to be either a subset or a superset (or neither) of the input. Theirs allows `...` again.
* No support for einsum, but `torch.matmul` handles batched matrix multiplication.
