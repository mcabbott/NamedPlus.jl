# NamedPlus.jl

This package exists to try out ideas for [NamedDims.jl](https://github.com/invenia/NamedDims.jl),
a lightweight package which attaches names to the dimensions/indices/axes of arrays.
The ways to use this are: 

1. It can be used to check a calculation which is written to work on ordinary arrays.
  The names should be propagated through to the answer, and any operation which combines
  incompatible dimension names should give an error. 

2. You can write some operations in terms of names not dimension number
  for clarity. Things like `A[μ=1, ν=2]` and `sum(A; dims=:μ)` already work.

3. Or you can try to write all operations working only on the names, to make results independent of
  the storage order of the data. 

Functions defined here like `contract(A,B; dims=:μ)` aim at the at 2nd or 3rd use.
`unname(A, (:α,:β))` which guarantees the order of the array (transposing only if necessary)
aims at the 3rd.

This package also tries to improve the handling wrapper types, and provides some convenience macros:

```julia
using NamedPlus, LinearAlgebra

# convenience macros
@namedef begin
    rand(3) => v{j}
    rand(2,3) => m{i,j}   # define m whose type includes (:i, :j)
    rand(2,3,4) => t{i,j,k}
    contract => *ⱼ{j}     # infix contraction funciton over :j
end;
w = @dropdims sum(m; dims=:i) # 3-element NamedDimsArray{(:j,),...

# wrapper types
d = Diagonal(v)        # 3×3 Diagonal{Float64,NamedDimsArray{(:j,), ...
d.names                # (:j, :j), using getproperty(::NamedUnion, ...)
unname(d)              # looks inside
diagonal(v, (:j, :j′)) # NamedDimsArray{(:j, :j′), ..., Diagonal{...

p = PermutedDimsArray(t, (3,1,2))
p.names                # (:k, :i, :j)
p == PermutedDimsArray(t, (:k,:i,:j)) # works too, same wrapper
t == canonise(p)       # unwraps
```

New functions / methods:

```julia
# contract(v, m; dims=:j) knows to transpose:
v *ⱼ m           # index i
m *ⱼ diagonal(v) # indices i,j

@code_warntype v *ⱼ m # fine! uses Contract{(:j,)}(v,m)
@code_warntype contract(v,m; dims=:j) # ::Any, hence slow

using OMEinsum # allows contraction with a 3-tensor
t *ⱼ m           # indices i,k
t *ⱼ diagonal(v) # indices i,k
t *ⱼ diagonal(v, (:j, :j′)) # indices i,k,j′

# rename, unname
mk = rename(m, :j => :k) # replace an index
tm = rand()<0.5 ? m : copy(transpose(m))
unname(tm, (:i, :j))     # un-named array, always 2 x 3, sometimes ::Transpose

prime(m, first) # names (:i′, :j)
prime(m, 2)     # (:i, :j′)
```

Adapting [PR#24](https://github.com/invenia/NamedDims.jl/pull/24) to make SVD work similarly...
not so sure this is the right idea, but `contract(U,S,V; dims=:svd)` needs something label to work on.
It would be easy to make `svd(m; dims)` control the order (i.e. which is `U`), 
but making `svd(m; name)` control the name of the new index would be harder. 

```julia
# svd uses label :svd, in order to have something to contract on
s = rand()<0.5 ? svd(m) : svd(transpose(m));
s[:i]       # could be s.U or s.Vt depending on order of m's indices
s[:j]       # always :j and :svd, in some order
s[:svd]     # always s.S

contract(s.U, s.S, s.V; dims=:svd) # contract three objects, leaving indices i & j

@namedef s[:j]{j,svd} => U # sure of having (j,svd) order, always size 3 x 2, sometimes ::Transpose
```

Something about this seems to break Revise.jl, BTW.
