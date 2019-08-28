module NamedPlus

using NamedDims, LinearAlgebra
using NamedDims: names

export NamedDims, NamedDimsArray, unname, dim

const CoVector = Union{Adjoint{<:Any, <:AbstractVector}, Transpose{<:Any, <:AbstractVector}}

#################### NAMES ####################

export dimnames

"""
    dimnames(x)
    dimnames(x, d) = dimnames(x)[d]
    x.names

The same as `NamedDims.names(x)`, minus the clash with `Base.names`.
And `names(x, d)` works like `size(x, d)`.
"""
dimnames(x) = NamedDims.names(x)

NamedDims.names(x, d::Int) = d <= ndims(x) ? NamedDims.names(x)[d] : :_
dimnames(x,d::Int) = NamedDims.names(x, d)

#################### WRAPPERS ####################
# replacing https://github.com/invenia/NamedDims.jl/pull/64

function Base.PermutedDimsArray(nda::NamedDimsArray{L,T,N}, perm::NTuple{N,Symbol}) where {L,T,N}
    numerical_perm = dim(nda, perm)
    PermutedDimsArray(nda, numerical_perm)
end

"""
`NamedMat{T,S}` is a union type for `NamedDimsArray` and wrappers containing this,
such as `Transpose` & `Diagonal`. The object always has `ndims(x)==2`,
but may involve a wrapped `NamedDimsArray` with `ndims(x.parent)==1`.
Type does not have `{L}` as that would not be equal to `names(x)`
"""
const NamedMat{T,S} = Union{
    NamedDimsArray{L,T,2,S} where {L},

    Diagonal{T,NamedDimsArray{L,T,1,S}} where {L},
    Transpose{T,NamedDimsArray{L,T,N,S}} where {L,N},
    Adjoint{T,NamedDimsArray{L,T,N,S}} where {L,N},

    Symmetric{T,NamedDimsArray{L,T,N,S}} where {L,N},
    Hermitian{T,NamedDimsArray{L,T,N,S}} where {L,N},
    UpperTriangular{T,NamedDimsArray{L,T,N,S}} where {L,N},
    UnitUpperTriangular{T,NamedDimsArray{L,T,N,S}} where {L,N},
    LowerTriangular{T,NamedDimsArray{L,T,N,S}} where {L,N},
    UnitLowerTriangular{T,NamedDimsArray{L,T,N,S}} where {L,N},
    Tridiagonal{T,NamedDimsArray{L,T,N,S}} where {L,N},
    SymTridiagonal{T,NamedDimsArray{L,T,N,S}} where {L,N},
    }

const NamedVec{T,S} = NamedDimsArray{L,T,1,S} where {L} # no 1D wrappers, just a name

const NamedVecOrMat{T,S} = Union{NamedVec, NamedMat}

"""
`NamedUnion{T,S}` is a union type for `NamedDimsArray{L,T,N,S}`
and wrappers containing this, such as `PermutedDimsArray`, `Diagonal`, `Transpose` etc.
"""
const NamedUnion{T,S} = Union{
    NamedDimsArray{L,T,N,S} where {L,N},
    NamedMat{T,S},
    PermutedDimsArray{T,N,P,Q,NamedDimsArray{L,T,N,S}} where {L,N,P,Q},
    }

Base.getproperty(x::NamedUnion{L}, s::Symbol) where {L} =
    s===:names ? dimnames(x) :
    getfield(x, s)

Base.getproperty(x::NamedDimsArray{L,T,N,S}, s::Symbol) where {L,T,N,S} =
    s===:parent ? parent(x) :
    getfield(x, s)

export diagonal

"""
    diagonal(m)

`diagonal` is to `Diagonal` about as `transpose` is to `Transpose`.
On an ordinary Vector there is no distinction, but on a NamedDimsArray it returns
`NamedDimsArray{T,2,Diagonal}` instead of `Diagonal{T,NamedDimsArray}`.
This has two independent names, by default the same as that of the vector,
but writing `diagonal(m, names)` they can be set.
"""
diagonal(x) = Diagonal(x)
diagonal(x::NamedDimsArray{L,T,1}) where {L,T} = NamedDimsArray{(L[1],L[1])}(Diagonal(x.data))
diagonal(x::NamedMat) where {L,T} = NamedDimsArray{dimnames(x)}(Diagonal(x.data))

diagonal(x::AbstractArray, s::Symbol) = diagonal(x, (s,s))
diagonal(x::AbstractArray, tup::Tuple{Symbol, Symbol}) =
    NamedDimsArray{tup}(Diagonal(x))
diagonal(x::NamedUnion, tup::Tuple{Symbol, Symbol}) =
    NamedDimsArray{tup}(Diagonal(unname(x)))

using TupleTools

NamedDims.names(::Type{Diagonal{T,NamedDimsArray{L,T,1,S}}}) where {L,T,S} = (L[1], L[1])

NamedDims.names(::Type{Transpose{T,NamedDimsArray{L,T,1,S}}}) where {L,T,S} = (:_, L[1])
NamedDims.names(::Type{Transpose{T,NamedDimsArray{L,T,2,S}}}) where {L,T,S} = (L[2], L[1])

NamedDims.names(::Type{Adjoint{T,NamedDimsArray{L,T,1,S}}}) where {L,T,S} = (:_, L[1])
NamedDims.names(::Type{Adjoint{T,NamedDimsArray{L,T,2,S}}}) where {L,T,S} = (L[2], L[1])

NamedDims.names(::Type{PermutedDimsArray{T,N,P,Q,NamedDimsArray{L,T,N,S}}}) where {L,T,N,S,P,Q} =
    TupleTools.permute(L, P)


NamedDims.unname(x::Diagonal{T,NamedDimsArray{L,T,N,S}}) where {L,T,N,S} = Diagonal(x.diag.data)
NamedDims.unname(x::Transpose{T,NamedDimsArray{L,T,N,S}}) where {L,T,N,S} = Transpose(x.parent.data)
NamedDims.unname(x::Adjoint{T,NamedDimsArray{L,T,N,S}}) where {L,T,N,S} = Adjoint(x.parent.data)
NamedDims.unname(x::PermutedDimsArray{T,N,P,Q,NamedDimsArray{L,T,N,S}}) where {L,T,N,S,P,Q} = PermutedDimsArray(x.parent.data, P)


# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#Special-matrices-1
for Wrap in (
    :Symmetric,
    :Hermitian,
    :UpperTriangular,
    :UnitUpperTriangular,
    :LowerTriangular,
    :UnitLowerTriangular,
    # :Tridiagonal,
    # :SymTridiagonal, # these are more like Diagonal, wrap vectors
    )
    @eval NamedDims.names(::Type{$Wrap{T,NamedDimsArray{L,T,2,S}}}) where {L,T,S} = L
    @eval NamedDims.unname(::Type{$Wrap{T,NamedDimsArray{L,T,2,S}}}) where {L,T,S} = $Wrap(parent(parent(x)))
end
# unname(Symmetric(NamedDimsArray(rand(3,3), (:a, :b))))

# can we see inside other types not yet loaded? This breaks everything...
# and parent(typeof(Diagonal(v))) is an error
# function NamedDims.names(x::AbstractArray)
#     p = parent(x)
#     typeof(p) === typeof(x) ? ntuple(_->:_, ndims(x)) : NamedDims.names(p)
# end

#################### SHOW ####################

function Base.summary(io::IO, x::NamedUnion)
    if ndims(x)==1
        print(io, length(x),"-element (",summary_pair(dimnames(x)[1],axes(x,1)),") ",typeof(x))
    else
        list = [summary_pair(na,ax) for (na,ax) in zip(dimnames(x), axes(x))]
        print(io, join(list," × "), " ",typeof(x))
    end
end

summary_pair(name::Symbol, axis) =
    axis===Base.OneTo(1) ? string(name,"=1") :
    first(axis)==1 ? string(name,"≤",length(axis)) :
    string(name,"∈",first(axis),":",maximum(axis))

#################### UNWRAPPING ####################
## https://github.com/invenia/NamedDims.jl/issues/65

"""
    unname(A::NamedDimsArray, names) -> AbstractArray

Returns the parent array if the given names match those of `A`,
otherwise a `transpose` or `PermutedDimsArray` view of the parent data.
To ensure a copy, call instead `parent(permutedims(A, names))`.
"""
function NamedDims.unname(nda::NamedUnion, names::NTuple{N, Symbol}) where {N}
    perm = dim(nda, names)
    if perm == ntuple(identity, N)
        return unname(nda)
    elseif perm == (2,1)
        return transpose(unname(nda))
    else
        return PermutedDimsArray(unname(nda), perm)
    end
end

#################### RENAME ####################

export rename

"""
    rename(A, names) = NamedDims.rename(A, names)

Discards `A`'s dimension names & replaces with the given ones.
Exactly equivalent to `NamedDimsArray(unname(A), names)` I think.

    rename(A, :i => :j)
    A′, B′ = rename(A, B, :i => :j, :j => :k)

Works a bit like `Base.replace` on index names.
If there are several rules, the first matching rule is applied to each index, not all in sequence.
Given several arrays `A, B`, it makes the same replacements for all, returning a tuple.
"""
rename(nda::NamedUnion, names::NTuple{N, Symbol} where N) = NamedDims.rename(nda, names)

function rename(nda::NamedUnion, pairs::Pair...)
    old = dimnames(nda)
    new = map(old) do i
        for p in pairs
            i == p.first && return p.second
        end
        return i
    end
    NamedDimsArray(unname(nda), new)
end

for n=2:10
    args = [:( $(Symbol("nda_",i))::NamedUnion ) for i=1:n ]
    vals = [:( rename($(Symbol("nda_",i)), pairs...) ) for i=1:n ]
    @eval rename($(args...), pairs::Pair...) = ($(vals...),)
end

#################### CANONICALISE ####################

export canonise

"""
    A′ = canonise(A::NamedDimsArray)

Re-arranges the index names of `A` to canonical order,
meaning that they now match the underlying storage,
removing lazy re-orderings such as `Transpose`.

This should not affect any operations which work on the index names,
but will confuse anything working on index positions.

`NamedDimsArray{L,T,2,Diagonal}` is unwrapped only if it has two equal names,
which `Diagonal{T,NamedVec}` always has, and thus is always unwrapped to a vector,
"""
canonise(x) = begin
    # @info "nothing to canonicalise?" typeof(x)
    x
end

# diagonal / Diagonal
canonise(x::Diagonal{T,<:NamedDimsArray{L,T,1}}) where {L,T} = x.diag
canon_names(::Type{Diagonal{T,<:NamedDimsArray{L,T,1}}}) where {L,T} = L

canonise(x::NamedDimsArray{L,T,2,<:Diagonal{T,<:AbstractArray{T,1}}}) where {L,T<:Number} =
    L[1] === L[2] ? NamedDimsArray{(L[1],)}(x.data.diag) : x
canon_names(::Type{NamedDimsArray{L,T,2,<:Diagonal{T,<:AbstractArray{T,1}}}}) where {L,T<:Number} =
    L[1] === L[2] ? (L[1],) : L # TODO make canon_names for all the rest? automate?


# PermutedDimsArray
canonise(x::PermutedDimsArray{T,N,P,Q,NamedDimsArray{L,T,N,S}}) where {L,T,N,S,P,Q} =
    NamedDimsArray{L}(x.parent.data)

# transpose / Transpose of a matrix (of numbers, to avoid recursion)
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,2}}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(x.data.parent)

canonise(x::Transpose{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Number} =
    NamedDimsArray{(L[2],L[1])}(x.parent.data)

# transpose / Transpose of a vector: drop :_ dimension
canonise(x::NamedDimsArray{L,T,2,<:Transpose{T,<:AbstractArray{T,1}}}) where {L,T<:Number} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(x.data.parent) : x

canonise(x::Transpose{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Number} = x.parent

# same for Adjoint but only on reals
canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,2}}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(x.data.parent)

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,2}}) where {L,T<:Real} =
    NamedDimsArray{(L[2],L[1])}(x.parent.data)

canonise(x::NamedDimsArray{L,T,2,<:Adjoint{T,<:AbstractArray{T,1}}}) where {L,T<:Real} =
    L[1] === :_ ? NamedDimsArray{(L[2],)}(x.data.parent) : x

canonise(x::Adjoint{T,<:NamedDimsArray{L,T,1}}) where {L,T<:Real} = x.parent


#################### PERMUTENAMES ####################

export permutenames

"""
    permutenames(A, names)
    permutenames(A, names, lazy=false)

This is a bit like `permutedims`, but does not copy the data to a new array in the given order,
and instead wraps it in `Transpose` or `PemutedDimsArray` if nedded.
This is the default `lazy=true` behaviour. Keyword `lazy=false` will copy only if needed
to avoid these wrappers. This is not exactly `permutedims(A, names)`, as that always copies.

For now requires `length(names) == ndims(canonise(A))`.
Perhaps it should allow longer lists, and insert trivial dimensions `:_` as needed?

Note that `canonise(A)` unwraps `Diagonal{...,Vector}` and `Transpose{...,Vector}`
to have just one index.
"""
function permutenames(A::NamedUnion, names::NTuple{N,Symbol} where N; lazy::Bool=true)
    B = canonise(A)
    perm = dim(B, names)
    if perm == ntuple(identity, ndims(B))
        return B
    elseif perm == (2,1)
        return lazy ? transpose(B) : copy(transpose(B))
    else
        C = lazy ? PermutedDimsArray(unname(B), perm) : permutedims(unname(B), perm)
        return NamedDimsArray{names}(C)
    end
end


#################### SVD ETC ####################
## https://github.com/invenia/NamedDims.jl/issues/12
## https://github.com/invenia/NamedDims.jl/pull/24

include("functions_linearalgebra.jl")

function Base.getproperty(fact::SVD{T, Tr, <:NamedDimsArray{L}}, d::Symbol) where {T, Tr, L}
    inner = getproperty(parent(fact), d)
    n1, n2 = L
    if d === :U
        return NamedDimsArray{(n1,:svd)}(inner)
    elseif d === :V
        return NamedDimsArray{(n2, :svd)}(inner) # V, Vt were backwards
    elseif d === :Vt
        return NamedDimsArray{(:svd, n2)}(inner)
    else # :S
        return NamedDimsArray{(:svd,)}(inner)
    end
end

# Piracy...
function Base.getindex(fact::SVD{T,Tr,<:NamedDimsArray{L}}, d::Symbol) where {T, Tr, L}
    n1, n2 = L
    if d === n1
        return fact.U
    elseif d === n2
        return fact.Vt
    elseif d === :svd
        return fact.S
    else
        error("expected symbol :$d to be either :svd or one of $L")
    end
end

#################### CONTRACTION ####################
## https://github.com/invenia/NamedDims.jl/pull/63

export contract

struct Contract{L} end

"""
    contract(A::NamedMat, B::NamedMat; dims)
    contract(C, D, E, ...; dims)

Matrix multiplication, contracting indices `dims` on `A` and `B`.
Or generalised contraction, contracting the same indices on on all `C`, `D`, `E` etc,
for which you need package `OMEinsum`.

A slightly awkward edge case here is diagonal matrices. If they have the same name twice,
say `names(B) == (j,j)`, then matrix multiplication treats this as a matrix,
giving another matrix, `(i,j) * (j,j) -> (j,j)`.
But generalised contraction canonicalises such a `B` to a vector,
so contraction with a 3-tensor sends `(i,j,k) * (j,j) -> (i,k)` with no `j` indices surviving.
Diagonal matrices with two different names, such as made by `diagonal(vec, names)`,
are always treated as matrices.
"""
contract(xs::NamedUnion...; dims) = make_contract(dims, xs...)
# Base.:*(x::NamedDimsArray{L,T,2}, xs::NamedUnion...; dims) where {L,T} = make_contract(dims, x, xs...)

make_contract(dims, xs...) =
    dims isa NTuple{N,Symbol} where {N} ? Contract{NamedDims.compile_time_return_hack(dims)}(xs...) :
    dims isa Vector{Symbol} ? Contract{Tuple(dims)}(xs...) :
    dims isa Symbol ? Contract{NamedDims.compile_time_return_hack((dims,))}(xs...) :
    error("contraction *(xs...; dims) must be over one or more symbols")

function Contract{dims}(x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,1}) where {dims, Lx,Tx, Ly,Ty}
    dims[1] == Lx[1] == Ly[1] && length(dims)==1 || throw_contract_dim_error(dims, x, y)
    return transpose(x) * y
end

function Contract{dims}(x::NamedMat, y::NamedVec) where {dims}
    Lx = dimnames(x)
    s = dims[1]
    length(dims) > 1 && throw_contract_dim_error(dims, x, y)
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return transpose(x) * y
    else
        throw_contract_dim_error(s, x, y)
    end
end

function Contract{dims}(x::NamedVec, y::NamedMat) where {dims}
    # return Contract{dims}(y, x) # this is wrong order, if elements of x & y don't commute
    return canonise(transpose(y) * x)
end

function Contract{dims}(x::NamedMat, y::NamedMat) where {dims}
    Lx, Ly = dimnames(x), dimnames(y)
    s = dims[1]
    length(dims) > 1 && throw_contract_dim_error(dims, x, y)
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return transpose(x) * y
    elseif s == Lx[2] == Ly[2]
        return x * transpose(y)
    elseif s == Lx[1] == Ly[2]
        return transpose(x) * transpose(y)
    else
        throw_contract_dim_error(s, x, y)
    end
end

function throw_contract_dim_error(s::Symbol, x, y)
    msg = "Cannot contract index :$s between arrays with indices $(names(x)) and $(names(x))"
    return throw(DimensionMismatch(msg))
end
function throw_contract_dim_error(dims::Tuple, x, y)
    msg = "Cannot contract indices $dims between arrays with $(names(x)) and $(names(x))"
    return throw(DimensionMismatch(msg))
end

using Requires

function __init__()
    @require OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922" include("omeinsum_compat.jl")
end

#=
#################### OLD MACROS ####################
## https://github.com/invenia/NamedDims.jl/pull/62

# export @named, @unname

using MacroTools

"""
    @named begin
        A′ = A[i,j,k]
        B′ = B[k]
        C[i,_,k]
        f(A′, C) .+ B′
    end

Convenience macro for adding dimension names, or asserting that they agree.
The same as calling `A′ = NamedDimsArray(A, (:i, :j, :k))`. Here it wil be asserted
that `C` is a 3-tensor whose dimension names (if any) are compatible with `(:i, :_, :k)`.
Other expressions like `f` are run as usual.

    @named begin
        *′ = *(i)
        *ᵢⱼ = *(i,j)
        /ⱼ = /(j)
    end

This defines a function `*′` which multiplies two `NamedDimsArray`s always along index `i`.
(You will need the package `OMEinsum` for 3-index and higher arrays, or to sum over multiple indices.)
The function may have any name, but decorations of `*` such as `*′` or `*ᵢⱼ` give an
infix operator which may be used `A′ *′ D`.
Similarly `/ⱼ` is like `/` but transposes as needed to act on shared index `j`.
(Soon!)
"""
macro named(ex)
    named_macro(ex)
end

function named_macro(input_ex)
    outex = MacroTools.prewalk(input_ex) do ex

        if @capture(ex, A_[ijk__])
            return :( NamedDims.NamedDimsArray{($(QuoteNode.(ijk)...),)}($A) )

        elseif @capture(ex, s_ = *(i_) )
            return :( $s(xs...) = NamedPlus.Contract{($(QuoteNode(i)),)}(xs...) )
        elseif @capture(ex, s_ = *(ijk__) )
            return :( $s(xs...) = NamedPlus.Contract{($(QuoteNode.(ijk)...),)}(xs...) )

        elseif @capture(ex, s_ = /(i_) )
            return :( $s(x,y) = NamedPlus.LinearSolve{$(QuoteNode(i))}(x,y) )

        end
        return ex
    end
    esc(outex)
end

"""
    @unname A[i,j,k]

Macro for removing dimension names. If `A == @named A[i,j,k]` then this returns `parent(A)`,
but if `A` has these names in another order, then it returns `permutedims(parent(A), (:i,:j,:k))`.
"""
macro unname(ex)
    unname_macro(ex)
end

function unname_macro(input_ex)
    outex = MacroTools.prewalk(input_ex) do ex
        if @capture(ex, A_[ijk__])
            stup = :( ($(QuoteNode.(ijk)...),) )
            return :( NamedDims.names($A) == $stup ? parent($A) : parent(permutedims($A, $stup)) )
        end

        return ex
    end
end

=#

#################### NEW MACRO ####################

export @namedef

using MacroTools

"""
    @namedef begin
        A => A′{i,j}
        B′{i,j} => B
        C′{i,j} => C′′{x,y}
        contract => *′{i}
    end
    A′ = @namedef A => {i,j}

Macro for adding and removing dimension names.
`A′ = NamedDimsArray{(:i, :j)}(A)` has the same data as `A` but its type contains `i,j`.
It is asserted that `B′` has names `i,j`, and this is unwrapped to array `B` in this order.
`C′′` is a re-named version of `C′`.
And `*′` is an infix contraction function along index `j`.
"""
macro namedef(ex)
    names_macro(ex)
end

function names_macro(input_ex)
    outex = quote end
    if input_ex.head == :block
        for ex in input_ex.args
            ex isa LineNumberNode && continue

            # Special words contract etc must come first
            if @capture(ex, contract => g_{ijk__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                push!(outex.args, :( $g(x...) = NamedPlus.Contract{$stup}(x...) ))

            elseif @capture(ex, solve => h_{ijk__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                push!(outex.args, :( $h(x...) = NamedPlus.LinearSolve{$stup}(x...) ))

            # Then conversion of arrays etc
            elseif @capture(ex, A_ => B_{ijk__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                push!(outex.args, :( $B = NamedDims.NamedDimsArray($A, $stup) ))

            elseif @capture(ex, C_{ijk__} => D_)
                stup = :( ($(QuoteNode.(ijk)...),) )
                push!(outex.args, :( $D = NamedDims.unname($C, $stup) ))

            elseif @capture(ex, E_{ijk__} => F_{xyz__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                stup2 = :( ($(QuoteNode.(xyz)...),) )
                push!(outex.args, :( $F = NamedDims.NamedDimsArray(NamedDims.unname($C, $stup), $stup2) ))

            else
                error("@names doesn't know what to do with $ex")
            end
        end
    else
        if @capture(input_ex, A_ => {ijk__})
            @gensym B
            return names_macro(quote
                $A => $B{$(ijk...)}
            end)
        else
            return names_macro(quote
                $input_ex
            end)
        end
    end
    esc(outex)
end

#################### DROPDIMS ####################

export @dropdims

using MacroTools

"""
    @dropdims sum(A; dims=1)

Macro which wraps such reductions in `dropdims(...; dims=1)`.
Allows `sum(A; dims=1) do x stuff end`,
and works on whole blocks of code like `@views`.
Does not handle other keywords, like `reduce(...; dims=..., init=...)`.
"""
macro dropdims(ex)
    _dropdims(ex)
end

function _dropdims(ex)
    out = MacroTools.postwalk(ex) do x
        if @capture(x, red_(args__, dims=d_)) || @capture(x, red_(args__; dims=d_))
            :( dropdims($x; dims=$d) )
        elseif @capture(x, dropdims(red_(args__, dims=d1_); dims=d2_) do z_ body_ end) ||
               @capture(x, dropdims(red_(args__; dims=d1_); dims=d2_) do z_ body_ end)
            :( dropdims($red($z -> $body, $(args...); dims=$d1); dims=$d2) )
        else
            x
        end
    end
    esc(out)
end

#################### PRIMES ####################

export prime

"""
    prime(x, d::Int)
    prime(x, first)
    prime(x, last)
    prime(x, i::Symbol) = rename(x, i => prime(i))

Add a unicode prime `′` to either the indicated index name, or to the given symbol.
Acting on symbols, `prime(s) == Symbol(s, '′')` but faster.
"""
prime(s::Symbol)::Symbol = _prime(Val(s))
# @btime (() -> prime(:a))() # shows 0 allocations
@generated function _prime(vals::Val{s}) where {s}
    QuoteNode(Symbol(s, Symbol('′')))
end

# @btime (() -> _prime((:i,:j,:k), Val(1)))() #  6ns, 1 allocation
_prime(tup::NTuple{N,Symbol}, ::Val{n}) where {N,n} =
    ntuple(i -> i==n ? prime(tup[i])::Symbol : tup[i], N)

using NamedPlus: NamedUnion

prime(x::NamedUnion, d::Int) = rename(x, _prime(NamedDims.names(x), Val(d)))
prime(x::NamedUnion, ::typeof(first)) = rename(x, _prime(NamedDims.names(x), Val(1)))
prime(x::NamedUnion, ::typeof(last)) = rename(x, _prime(NamedDims.names(x), Val(ndims(x))))

prime(x::NamedUnion, s::Symbol) = rename(x, s => prime(s))

"""
    :x' == :x′

`adjoint(::Symbol)` adds unicode prime `′` to the end.
"""
Base.adjoint(s::Symbol) = prime(s)




#################### THE END ####################

end # module
