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

#################### DIAGONAL ####################

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
diagonal(x::NamedMat) where {L,T} = NamedDimsArray{names(x)}(Diagonal(x.data))

diagonal(x::AbstractArray, s::Symbol) = diagonal(x, (s,s))
diagonal(x::AbstractArray, tup::Tuple{Symbol, Symbol}) =
    NamedDimsArray{tup}(Diagonal(x))
diagonal(x::NamedUnion, tup::Tuple{Symbol, Symbol}) =
    NamedDimsArray{tup}(Diagonal(unname(x)))


#################### SIMILAR ####################

"""
    similar(A::NamedUnion, names)
    similar(A,B,C, names)

Creates a new `NamedDimsArray` like `A`, with the given names,
and sizes read off from the matching dimensions of `A`, or `B`, etc.
"""
Base.similar(xyz::NTuple{M,NamedUnion}, names::NTuple{N,Symbol}) where {M,N} =
    NamedDimsArray{names}(similar(unname(first(xyz)), map(i -> first_size(i, xyz...), names)))

# Same with eltype T specified
Base.similar(xyz::NTuple{M,NamedUnion}, T::Type, names::NTuple{N,Symbol}) where {M,N} =
    NamedDimsArray{names}(similar(unname(first(xyz)), T, map(i -> first_size(i, xyz...), names)))

first_size(i, x, yz...) = Base.sym_in(i, names(x)) ? size(x,i) : first_size(i, yz...)
first_size(i::Symbol) = error("none of the supplied arrays had name :$i")

for m=1:10
    syms = [ Symbol(:x_,n) for n in 1:m ]
    args = [ :($(syms[n])::NamedUnion) for n in 1:m ]
    @eval begin
        Base.similar($(args...), names::NTuple{N,Symbol}) where {N} =
            similar(($(syms...),), names)

        Base.similar($(args...), i::Symbol, names::Symbol...) where {N} =
            similar(($(syms...),), (i,names...))

        # Same with eltype T specified
        Base.similar($(args...), T::Type, names::NTuple{N,Symbol}) where {N} =
            similar(($(syms...),), T, names)

        Base.similar($(args...), T::Type, i::Symbol, names::Symbol...) where {N} =
            similar(($(syms...),), T, (i,names...))
    end
end

####################
