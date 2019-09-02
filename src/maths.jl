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
    Lx, Ly = names(x), names(y)
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
    return canonise(Contract{dims}(transpose(x),y))
end

function Contract{dims}(x::NamedMat, y::NamedMat) where {dims}
    Lx, Ly = names(x), names(y)
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

####################
