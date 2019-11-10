# NamedPlus.jl

[![Build Status](https://travis-ci.org/mcabbott/NamedPlus.jl.svg?branch=master)](https://travis-ci.org/mcabbott/NamedPlus.jl)

This package provides some utilities for working with arrays by dimension name. 
The names are those provided by [NamedDims.jl](https://github.com/invenia/NamedDims.jl),
which are simply a tuple of symbols (like those of a NamedTuple).
The overhead of keeping track of these names should happen entirely at compile-time
(or parse-time for the macro).

## `@named` macro

The simplest function of this macro is a shorthand for creating `NamedDimsArray`s,
using the notation `M{i,j}` to indicate the names (which are part of the type after all): 
```julia
@named M{i,j} = rand(Int8, 3,4)
names(M) == (:i, :j)
```

Second, it also applies names to arrays created by comprehensions, 
by reading off the iteration variable:
```julia
@named begin 
    G = [sqrt(k) for k in 1:5]
    R = [n//d for n in 1:3, d in 1:7]
end
names(G) == (:k,)
names(R) == (:n, :d)
```

And third, the macro will automatically re-arrange dimensions for broadcasting, 
to match a given target:
```julia
@named V{j} = rand(Int8, 4)
W = M .+ transpose(V)  # by default names are only used to check alignment

@named begin
    W{i,j} = M .+ V  # automatically transposes, etc.
    Z{h,i,j,k} = M .+ V ./ G
end;
names(Z) == (:_, :i, :j, :k)
size(Z) == (1, 3, 4, 5)
```

You can also access the same lazy re-orientation as a function `align`,
which takes either the target names, or an array which has them, as the second argument:
```julia
V′ = align(V, (:i,:j))
names(V′) = (:_, :j)

V′′ = align(V, Z)
names(V′′) = (:_, :_, :j, :_)
```

## `contract`

This provides generalised matrix multiplication, according to dimension name.
It can be used via this function:

```julia
@named begin # create some example arrays
  AB{a,b} = randn(3,4)
  CB{c,b} = randn(5,4)
  ABD{a,b,d} = randn(3,4,6)
end;

AC = contract(AB, CB, :b);
AC == AB * transpose(CB)

transpose(AC) == contract(CB, AB, :b)
```

There are other ways, using `@named`:
```julia
AC == @named AB *{b} CB  # won't work inside begin ... end

@named *ⱼ = *{b}  # define a function, which works infix
AC == AB *ⱼ CB
```

Contraction of things with more than two indices requires TensorOperations.jl. 
The `@named` macro interacts with `@tensor` to re-arrange indices if needed, 
and to name output arrays:
```julia
using TensorOperations

@named X = ABD *{b} BC

@named @tensor Y[a,c,d] := ABD[a,b,d] * BC[b,c]
names(Y) = (:a, :c, :d)
```

## `split` & `join`

These functions allow you to do reshaping operations which change the number of dimensions, 
without caring about their order. 
The trivial example is to reshape a matrix into a vector:
```julia
@named M{i,j} = rand(Int8, 2,3)

V = join(M, (:i, :j) => :ij)  # V.data == vec(M.data)
names(V) == (:ij,)
size(V) == (6,)
```

To reverse this, it is necessary to provide the dimensions of the final matrix:
```julia
M′ = split(V, :ij => (:i, :j), (2,3))  # M′.data == reshape(V.data, 2,3)
names(M′) == (:ij,)
size(M′) == (2,3)
```

Here `M`, `V` and `M′` are all views of the same memory. However, this is not guaranteed by
`join`, because a lazy view combining non-adjacent dimensions will be very slow. 
In such cases, it will copy the array:
```julia
@named T{i,j,k} = rand(Int8, 2,3,4)

N = join(T, (:i, :k) => :ik)  # N.data == reshape(permutedims(T.data, (1,3,2)), 8,3)
```

Note also that to split dimension `:ik` again, 
we should provide only the dimensions of those two indices:
```julia
T′ = split(N, :ik => (:i, :k), (2,4))
# T′ == split(N, :ik => (:i, :k), T) # get dimensions from T, seems broken

names(T′) == (:j, :i, :k)
permutedims(T′, (:i,:j,:k)) == T
```

## `rename` & `prime`

This method of `rename` is a bit like `replace` for strings:
```julia
@named T{i,j,k} = ones(2,3,4)

T′ = rename(T, :j => :Jay)
names(T′) == (:i, :Jay, :k)
```

And `prime` adds primes!
```julia
M′′ = prime(M, :j)
names(M′′) == (:i, :j′)
```

## `plot`

Also plot recipes.

