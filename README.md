# NamedPlus.jl

[![Build Status](https://travis-ci.org/mcabbott/NamedPlus.jl.svg?branch=master)](https://travis-ci.org/mcabbott/NamedPlus.jl)

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 
While that package is fairly minimal (and focused on providing a type with great performance), 
this one defines lots of useful functions. Some of them are only defined when other packages 
they need are loaded. Here's what works in `v0.0.1`:

Some convenient ways add names (exports `@named`, `named`):
```julia
@named begin
    m{i,j} = rand(Int8, 3,4)             # create a matrix whose type has (:i,:j)
    g = [n^i for n in 1:20, i in 1:3]    # read names (:n,:i) from generator's variables
end
ones(i=1, j=4) .+ rand(Int8, i=3)        # names from keywords, needs rand(Type, i=...)

using EllipsisNotation
a_z = named(rand(4,1,1,2), :a, .., :z)   # adds names, or refines existing ones
rename(m, :i => :z')                     # renames just :i, to :z' == :z′
```

Some functions controlled by them:
```julia
t = split(g, :n => (j=4, k=5))           # just reshape, new size (4,5,3)
join(t, (:i, :k) => :χ)                  # copy if non-adjacent, size (4,15)

dropdims(a_z)                            # defaults to :_, and kills all of them
transpose(a_z, :a, :z)                   # permutes (4,2,3,1)
```

A hack to make lots of code propagate names (`NamedInt`):
```julia
d,k = size(m); @show d                   # NamedInt, which exists for:
z = zeros(d,d')                          # ones, fill, rand, etc
z .= [sqrt(i) for i in 1:d, i′ in 1:d']  # comprehensions propagate names from (1:d)
reshape(g, k,:,d) .+ g[end, d]           # reshape propagate via sizes

using Einsum, TensorCast
@einsum mz[i,k] := m[i,j] * z[i,k]       # works because of Array{}(undef, NamedInt...)
@cast tm[i⊗j,k] := t[j,k,i] + m[i,j]     # works because of reshape(A, NamedInt)
```

Some automatic re-ordering of dimensions (`align`, `align_sum!`, `align_prod!`):
```julia
align(m, (:j, :k, :i))                   # lazy generalised permutedims
@named q{i,j,k} = m .+ t                 # used for auto-permuted broadcasting
align(m, t) .+ t                         # or to manually fix things up

align_sum!(Int.(m), t)                   # reduce (:j, :k, :i) into (:i, :j)
```

Including for matrix multiplication (`mul`, `*ᵃ`, `contract`, `batchmul`):
```julia
m *ᵃ z == mul(m, z, :i) == m' * z        # matrix multiplication on shared index
g *ᵃ m == (m *ᵃ g)'

using TensorOperations                   # named inputs re-arranged via Strided
@named @tensor p[j,i′] := m[i,j] * z[i,i′]
contract(m, t)                           # shared indices i & j, leaving only k

using Zygote                             # contract defines a gradient
gradient(m -> sum(contract(m,t)[1]), m)[1]

using OMEinsum
contract(m, t, z)                        # sum over shared :i, leaving (:j, :k, :i′)
*ᵇ = batchmul(:k)                        # batch index :k,
t *ᵇ rename(t, :i => :i')                # sum over shared :j, leaving (:i, :i′, :k)
```

Some other bits have moved to [AxisRanges.jl](https://github.com/mcabbott/AxisRanges.jl).
If both packages are loaded:

```julia
using NamedPlus, AxisRanges, Plots
@named [n^i for n in 1:2:40, i in 2:4]   # has custom ranges
scatter(ans, yaxis=:log10)               # labels axes & series
```

Compared to Pytorch's [new named tensors](https://pytorch.org/docs/stable/named_tensor.html):

* `refine_names` ⟶ `named`, except with `..` instead of `...`.
* `unflatten` ⟶ `split` exactly, and `flatten` ⟶ `join`, except that for them "All of dims must be consecutive in order" while mine permutes if required.
* `.align_to` and `.align_as` ⟶ `align`, mine allows the target to be either a subset or a superset (or neither) of the input. Theirs allows `...` again.
* No support for einsum, but `torch.matmul` handles batched matrix multiplication.
