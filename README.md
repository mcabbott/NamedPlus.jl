# NamedPlus.jl

[![Build Status](https://travis-ci.org/mcabbott/NamedPlus.jl.svg?branch=master)](https://travis-ci.org/mcabbott/NamedPlus.jl)

This package exists to try out ideas for (or for use with) [NamedDims.jl](https://github.com/invenia/NamedDims.jl)
to see what's useful. `NamedDims` is a lightweight package which attaches names to the 
dimensions/indices/axes of arrays. 

This can be used to check a calculation which is written to work on ordinary arrays.
The names should be propagated through to the answer, and this happens at compile-time.
Any operation which combines incompatible dimension names should give an error.
Many operations, including broadcasting, work like this.

You can also write some operations in terms of names not dimension number
for clarity. Things like `A[μ=1, ν=2]` and `sum(A; dims=:μ)` already work.

Or, going further, you can try to write all operations working only on the names, 
to make results independent of the storage order of the data.
The first goal here is mostly to push in this third direction. 

The second goal is to see how well named dimensions can be made to work with other wrappers,
with some combination of union types & trait-like functions.

Here's what works: 

```julia
using NamedPlus, LinearAlgebra

# ===== convenience
@named begin
    v{j} = rand(3)
    m{i,j} = rand(2,3)    # define m whose type includes (:i, :j)
    t{i,j,k} = rand(2,3,4)
    z{z} = randn(26)
end;

similar(t, Int, :k)     # length == size(t, :k)
similar(m, z, :i, :z)   # names (:i, :z), size (2, 26)

# ===== broadcasting
w = m ./ v'                   # these indices line up, but m ./ v is an error
@named z{i,k,j} = t .+ m ./ v # these get automatically aligned
names(z) == (:i, :k, :j)

m′ = permutenames(m, (:i, :k, :j)) # by calling this, which wraps GapView{T,3,(1,0,2),(1,3),...
names(m′) == (:i, :_, :j)

# ===== contract(v, m; dims=:j) should insert transposes
m * v
m' ⊙ t # operator to contract neibouring indices, like python's @

@named *ⱼ = contract{j} # define infix contraction funciton over :j
v *ⱼ m           # index i
m *ⱼ diagonal(v) # indices i,j

@code_warntype v *ⱼ m # fine! uses Contract{(:j,)}(v,m)
@code_warntype contract(v,m; dims=:j) # ::Any, hence slow
@btime ((v,m) -> contract(v,m; dims=:j))($v, $m) # 5.4 μs, vs 155 ns

using OMEinsum # allows contraction with a 3-tensor
t *ⱼ m           # indices i,k
t *ⱼ diagonal(v) # indices i,k
t *ⱼ diagonal(v, (:j, :j′)) # indices i,k,j′

# ===== wrapper types
d = Diagonal(v)        # 3×3 Diagonal{Float64,NamedDimsArray{(:j,), ...
names(d)               # (:j, :j)
nameless(d)            # looks inside
diagonal(v, (:j, :j′)) # NamedDimsArray{(:j, :j′), ..., Diagonal{...
diagonal(v)[j=2]       # fixes both indices, not yet for Diagonal(v)

p = PermutedDimsArray(t, (3,1,2))
names(p)               # (:k, :i, :j)
summary(p)             # "k≤4 × i≤2 × j≤3 PermutedDimsArray{...
p == PermutedDimsArray(t, (:k,:i,:j)) # works too, same wrapper
t == canonise(p)       # unwraps

# ===== reshaping
join(t, (:i,:j) => :ij) # (:ij, :k) size (6, 4)
t′ = join(t, :i,:k)     # (:j, Symbol("i⊗k")) size (3, 8)

t′′ = split(t′, (:i,:k), (2,4)) # (:j, :i, :k) size (3, 2, 4)
t′′[1,1,1] = 99; t′ # changes t′ but not t, as non-adjacent join needed permutedims

# ===== rename, unname
mk = rename(m, :j => :k) # replace an index
tm = rand()<0.5 ? m : copy(transpose(m))
unname(tm, (:i, :j))     # un-named array, always 2 x 3, sometimes ::Transpose

prime(m, first) # names (:i′, :j)
prime(t, 2)     # (:i, :j′, :k)

# ===== reduction
@dropdims sum(t, dims=:j) # (:i, :k)
@named sum(m, dims={j})     # wraps functions ignorant of names
@named sum(m, dropdims={j}) # ... and wraps that in dropdims

using Distances # not extended by NamedDims
@named pairwise(Euclidean(), m, dims={j}) # is this what we want?
# @named D{i,i′} = pairwise(Euclidean(), m, dims={j}) # weird errors...

# ===== tensor notation
using TensorOperations # macro unwraps, and permutes if needed using Strided 
@named @tensor vk[k] := t[i,j,k] * tm[i,j]     # create NamedDimsArray{(:k,)
@named @tensor vk[k] = t[i,j,k] * tm[i,j]; vk  # write into vk, wrong return type :(

using OMEinsum # overloading its function, @named @ein not yet! 
@ein vk[k] := t[iii,j,k] * m[iii,j] # checks for clashes but ignores :iii

# ===== svd uses label :svd
s = rand()<0.5 ? svd(m) : svd(transpose(m));
s[:i]       # could be s.U or s.Vt depending on order of m's indices
s[:j]       # always :j and :svd, in some order
s[:svd]     # always s.S

contract(s.U, s.S, s.V; dims=:svd) # contract three objects, leaving indices i & j ## BROKEN

@named U = s[:j]{j,svd} # sure of having (j,svd) order, always size 3 x 2, sometimes ::Transpose
```

SVD is adapted from [PR#24](https://github.com/invenia/NamedDims.jl/pull/24). 
Not so sure this is the right idea, but `contract(U,S,V; dims=:svd)` needs a name to work on.
It would be easy to make `svd(m; dims)` control the order (i.e. which is `U`), 
but making `svd(m; name)` control the name of the new index would be harder. 

Finally there's also a draft of some ideas for attaching ranges to axes in [src/ranges.jl](src/ranges.jl).
This is done by adding an another independent wrapper type, which ought to commute with `NamedDims`.
You can index by the ranges using round brackets instead. Here's what works:

```julia
include("ranges.jl")

R = Wrap(rand(1:99, 3,4), ['a', 'b', 'c'], 10:10:40)
N = Wrap(rand(1:99, 3,4), obs = ['a', 'b', 'c'], iter = 10:10:40)

R('c', 40) == R[3, 4]
N('c', 40) == N[3, 4]

N(obs='a', iter=40) == N[obs=1, iter=4]
N(obs='a') == N('a') == N[1,:]

getnames(Transpose(N)) # unwraps, (:iter, :obs)
getranges(Transpose(N))

# ===== selectors
N(iter=Near(12.5))
N(iter=Between(7,23))

R('a', Index[2]) # back to square brackets

# ===== mutation
V = Wrap([3,5,7,11], μ=10:10:40)
push!(V, 13) # now μ ∈ 10:10:50

# ===== ranges can be any AbstractArray
using AcceleratedArrays
str = [string(gensym()) for _=1:20];
s13 = str[13]
S = Wrap((1:20) .+ im, s=str)
A = Wrap((1:20) .+ im, s=accelerate(str, UniqueHashIndex))

S(s = s13) == A(s13)  # uses findall(isequal(s1), s)
A(s = All(s13))       # uses findall(isequal(s1), s), which gets accelerated
```

Links:
* Older packages [AxisArrays](https://github.com/JuliaArrays/AxisArrays.jl) and 
  [NamedArrays](https://github.com/davidavdav/NamedArrays.jl),
  also [DimArrays](https://github.com/mcabbott/DimArrays.jl), 
  and [AxisArrayPlots](https://github.com/jw3126/AxisArrayPlots.jl) .
  And [LabelledArrays](https://github.com/JuliaDiffEq/LabelledArrays.jl) too!
* Discussion at [AxisArraysFuture](https://github.com/JuliaCollections/AxisArraysFuture/issues/1), and [AxisArrays#84](https://github.com/JuliaArrays/AxisArrays.jl/issues/84). 
* New packages: [NamedDims](https://github.com/invenia/NamedDims.jl) used here.
  Most like AxisArrays is [DimensionalData](https://github.com/rafaqz/DimensionalData.jl).
  `RangeWrap` is a bit like [IndexedDims](https://github.com/invenia/IndexedDims.jl),
  which uses [AcceleratedArrays](https://github.com/andyferris/AcceleratedArrays.jl) for ranges.
  Also [AbstractIndices](https://github.com/Tokazama/AbstractIndices.jl).
* Python's [xarray](http://xarray.pydata.org/en/stable/), Harvard NLP's [NamedTensor](http://nlp.seas.harvard.edu/NamedTensor). 

