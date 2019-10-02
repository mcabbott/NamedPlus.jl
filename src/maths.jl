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

"""
    *(s::Symbol, A::NamedDimsArray, B::NamedDimsArray)

Matrix multiplication by summing over the specified index `s`,
by evaluating something like `transpose(A) * B` if required.
Works only on matrices & vectors, not higher-dimensional arrays.
For two vectors it gives a scalar result.

Allows matrices with the same name twice.
When the elements are not just numbers (such as with arrays of matrices),
it uses `permutedims` to keep recursive `transpose` from acting on elements.
And it never changes the order of multiplication,
elements of `A` always left-multiply elements of `B`.

```
julia> A = NamedDimsArray(rand(2,3), (:i, :j));

julia> B = NamedDimsArray(rand(4,3), (:k, :j));

julia> *(:j, A, B) |> summary
"2×4 NamedDimsArray{(:i, :k),Float64,2,Array{Float64,2}}"

julia> *ⱼ(x,y) = *(:j,x,y)   # define infix function
*ⱼ (generic function with 1 method)

julia> B *ⱼ A |> summary
"4×2 NamedDimsArray{(:k, :i),Float64,2,Array{Float64,2}}"
```
"""
function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,1}) where {Lx,Tx,Ly,Ty}
    s == Lx[1] == Ly[1] || throw_contract_dim_error(s, x, y)
    if Tx <: Number
        return transpose(x) * y
    else
        return first(permutedims(x) * y)
    end
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,1}) where {Lx,Tx,Ly,Ty}
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return shallow_transpose(x) * y
    else
        throw_contract_dim_error(s, x, y)
    end
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,2}) where {Lx,Tx,Ly,Ty}
    dropdims(*(s, shallow_transpose(x), y), dims=1)
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,2}) where {Lx,Tx,Ly,Ty}
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return shallow_transpose(x) * y
    elseif s == Lx[2] == Ly[2]
        return x * shallow_transpose(y)
    elseif s == Lx[1] == Ly[2]
        return shallow_transpose(x) * shallow_transpose(y)
    else
        throw_contract_dim_error(s, x, y)
    end
end

function throw_contract_dim_error(s::Symbol, x, y)
    msg = "Cannot contract index :$s between arrays with indices $(names(x)) and $(names(x))"
    return throw(DimensionMismatch(msg))
end

shallow_transpose(x::AbstractArray{<:Number}) = transpose(x)
shallow_transpose(x::AbstractArray) = permutedims(x)


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

using Requires

function __init__()
    @require OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922" include("omeinsum_compat.jl")
end

# These aren't really needed for @generated version, but are for older one.
# Moved here to avoid world-age issues?

tuple_filter(f, t::Tuple) = _filter(f, t, ())
@inline function _filter(f, t::Tuple, r::Tuple)
    if f(first(t))
        return _filter(f, Base.tail(t), (r..., first(t)))
    else
        return _filter(f, Base.tail(t), r)
    end
end
_filter(f, t::Tuple{}, r::Tuple) = r

tuple_unique(t::Tuple) = _unique(t, ())
@inline function _unique(t::Tuple, r::Tuple)
    if first(t) in r
        return _unique(Base.tail(t), r)
    else
        return _unique(Base.tail(t), (r..., first(t)))
    end
end
_unique(t::Tuple{}, r::Tuple) = r

#################### PYTHONESQUE TIMES ####################

export ⊙

using NamedDims: valid_matmul_dims, throw_matrix_dim_error

"""
    A ⊙ B    # \\odot

Generalised matrix multiplication: contracts the last index of `A` with the first index of `B`.
Left-associative `A⊙B⊙C = (A⊙B)⊙C` like `*`.
"""
function ⊙(A::AbstractArray, B::AbstractArray)
    n = size(A, ndims(A))
    @assert size(B,1) == n "⊙ needs matching sizes on dimensions which touch"
    out = reshape(reshape(unname(A),:,n) * reshape(unname(B),n,:), _osizes(A,B)...)
    if A isa NamedUnion || B isa NamedUnion
        return NamedDimsArray{_onames(A,B)}(out)
    else
        return out
    end
end

_osizes(A::AbstractArray{T,N}, B::AbstractArray{S,M}) where {T,N,S,M} =
    ntuple(i -> i<N ? size(A, i) : size(B, i-N+2), Val(N+M-2))

function _onames(A::NamedUnion, B::NamedUnion)
    valid_matmul_dims(names(A), names(B)) || throw_matrix_dim_error(names(A), names(B))
    ntuple(i -> i<ndims(A) ? names(A, i) : names(B, i-ndims(A)+2), Val(ndims(A)+ndims(B)-2))
end
_onames(A::AbstractArray, B::NamedUnion) =
    ntuple(i -> i<ndims(A) ? :_ : names(B, i-ndims(A)+2), Val(ndims(A)+ndims(B)-2))
_onames(A::NamedUnion, B::AbstractArray) =
    ntuple(i -> i<ndims(A) ? names(A, i) : :_, Val(ndims(A)+ndims(B)-2))

⊙(A::AbstractMatrix, B::AbstractMatrix) = A*B
⊙(A::AbstractMatrix, B::AbstractVector) = A*B
⊙(A::AbstractVector, B::AbstractVector) = transpose(A)*B

⊙(A::AbstractArray, B::Number) = A*B
⊙(A::Number, B::AbstractArray) = A*B
⊙(A::Number, B::Number) = A*B


#################### SOFTMAX ####################
## https://github.com/FluxML/NNlib.jl/issues/77

function softmax1(xs::AbstractArray{T}; dims=1) where {T}
    max = maximum(xs, dims=dims)
    out = exp.(xs .- max) ./ sum(exp.(xs .- max), dims=dims)
end

function softmax2(xs::AbstractArray{T}; dims=1) where {T}
    temp = maximum(xs, dims=dims)
    out = exp.(xs .- temp)
    out ./= sum!(temp, out)
end

function softmax3(xs::AbstractArray{T}; dims=1) where {T}
    max = maximum(xs, dims=dims)
    out = exp.(xs .- max)
    out ./ sum(out, dims=dims)
end

####################
