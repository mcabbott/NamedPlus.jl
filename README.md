# NamedPlus.jl

This package exists to try out ideas for [NamedDims.jl](https://github.com/invenia/NamedDims.jl),
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
The goal here is mostly to push in this third direction. 

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
unname(d)              # looks inside
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

I hope it's obvious that none of this is well-tested, or reliable in any way! 
Some parts are slow, too.
