# NamedPlus.jl

[![Github CI](https://img.shields.io/github/workflow/status/mcabbott/NamedPlus.jl/CI?logo=github)](https://github.com/mcabbott/NamedPlus.jl/actions)
[![Tag Version](https://img.shields.io/github/v/tag/mcabbott/NamedPlus.jl?color=orange&logo=github)](https://github.com/mcabbott/NamedPlus.jl/releases)
[![Docstrings](https://img.shields.io/badge/julia-docstrings-blue.svg)](https://juliahub.com/docs/NamedPlus/)

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 
While that package is fairly minimal (and focused on providing a type with great performance), 
this one defines lots of useful functions. Some of them are only defined when other packages 
they need are loaded. Here's what works in `v0.0.1`:

Some convenient ways add names (exports `named`, `@named`, `nameless`):
```julia
@pirate Base
m = rand(Int8; i=3, j=4)                 # names from keywords, needs rand(Type, i=...)
m .+ ones(_=1, j=4, k=2)                 # ones(), zeros(), and fill() all work.

m .- named(parent(m), :i, :j)            # adds names, or refines existing ones, 
a_z = named(rand(4,1,1,2), :a, .., :z)   # use .. (from EllipsisNotation) to skip some.

@named g = [n^i for n in 1:20, i in 1:3] # read names (:n,:i) from generator's variables

rename(m, :i => :z')                     # renames just :i, to :z' == :z′
nameless(m, (:j, :i)) === transpose(m)   # also @named mt = m{j,i} 
```

Some functions controlled by them:
```julia
t = split(g, :n => (j=4, k=5))           # just reshape, new size (4,5,3),
join(t, (:i, :k) => :χ)                  # copy if non-adjacent, size (4,15).

dropdims(a_z)                            # defaults to :_, and kills all of them
transpose(a_z, :a, :z)                   # permutes (4,2,3,1)
```

A hack to make lots of code propagate names (`NamedInt`):
```julia
d,k = size(m); @show d                   # NamedInt, which exists for:
z = zeros(d,d')                          # ones, fill, rand, etc
z .= [sqrt(i) for i in 1:d, i′ in 1:d']  # comprehensions propagate names from (1:d)
reshape(g, k,:,d) .+ g[end, d]           # reshape propagate via sizes

using Einsum, TensorCast                 # These packages dont't know about names at all,
@einsum mz[i,k] := m[i,j] * z[i,k]       # works because of Array{}(undef, NamedInt...)
@cast tm[i⊗j,k] := t[j,k,i] + m[i,j]     # works because of reshape(A, NamedInt)
```

Some automatic re-ordering of dimensions (`align`, `align_sum!`, `align_prod!`):
```julia
align(m, (:j, :k, :i))                   # lazy generalised permutedims, (:j, :_, :i)
@named q{i,j,k} = m .+ t                 # used for auto-permuted broadcasting
align(m, t) .+ t                         # or to manually fix things up

align_sum!(Int.(m), t)                   # reduce (:j, :k, :i) into (:i, :j)
```

Including for matrix multiplication (`mul`, `*ᵃ`, `contract`, `batchmul`):
```julia
m *ᵃ z == mul(m, z, :i) == m' * z        # matrix multiplication on shared index,
g *ᵃ m == (m *ᵃ g)'                      # typed *\^a tab.

using TensorOperations
contract(m, t)                           # shared indices i & j, leaving only k
m ⊙ᵃ t == t ⊙ᵃ m                         # infix version, \odot\^a tab
@named @tensor p[j,i′] := m[i,j] * z[i,i′] # named inputs re-arranged via Strided

using OMEinsum
contract(m, t, z)                        # sum over shared :i, leaving (:j, :k, :i′)
const *ᵇ = batchmul(:k)                  # batch index :k,
t *ᵇ rename(t, :i => :i')                # sum over shared :j, leaving (:i, :i′, :k)

using Zygote                             
gradient(m -> sum(contract(m,t)[1]), m)[1] # contract defines a gradient
gradient(t -> sum(t *ᵇ q), t)[1]         # OMEinsum defines this gradient
```

Some other bits have moved to [AxisKeys.jl](https://github.com/mcabbott/AxisKeys.jl).
If both packages are loaded:

```julia
using NamedPlus, AxisKeys, Plots
@named [n^i for n in 1:2:40, i in 2:4]   # has custom ranges
scatter(ans, yaxis=:log10)               # labels axes & series
```

While the functions in [NamedDims.jl](https://github.com/invenia/NamedDims.jl) try very hard 
to be zero-cost (by working hard to exploit constant propagation), this is not true here.
In particluar `split`, `join`, `align`, `rename` will cost around 1μs.
(Perhaps if useful they can be made faster.) 
But `mul` and `*ᵃ`, and aligned broadcasting via `@named`, should be nearly free, perhaps 5ns.

Compared to Pytorch's [new named tensors](https://pytorch.org/docs/stable/named_tensor.html):

* `refine_names` ⟶ `named`, except with `..` instead of `...`.
* `unflatten` ⟶ `split` exactly, and `flatten` ⟶ `join`, except that for them "All of dims must be consecutive in order" while mine permutes if required.
* `.align_to` and `.align_as` ⟶ `align`, mine allows the target to be either a subset or a superset (or neither) of the input. Theirs allows `...` again.
* No support for einsum, but `torch.matmul` handles batched matrix multiplication.
