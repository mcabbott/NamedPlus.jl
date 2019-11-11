
#################### ZERO, ONES ####################

zero_doc = """
    zero(; r=2)
    ones(Int8; r=2, c=3)
    fill(3.14; c=3)

These are piratically overloaded to make `NamedDimsArray`s.
The zero-dimensional methods like `fill(3)` should still work.
"""
@doc zero_doc
Base.zeros(; kw...) = zeros(Float64; kw...)

function Base.zeros(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(zeros(T, kw.data...), kw.itr)
    else
        fill(zero(T), ())
    end
end

@doc zero_doc
Base.ones(; kw...) = ones(Float64; kw...)

function Base.ones(T::Type; kw...)
    if length(kw) >= 1
        NamedDimsArray(ones(T, kw.data...), kw.itr)
    else
        fill(one(T), ())
    end
end

@doc zero_doc
function Base.fill(v; kw...)
    if length(kw) >= 1
        NamedDimsArray(fill(v, Tuple(kw.data)), kw.itr)
    else
        fill(v, ())
    end
end


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
diagonal(x::NamedDimsArray{L,T,2}) where {L,T} = NamedDimsArray{L}(Diagonal(x.data))

diagonal(x::AbstractArray, s::Symbol) = diagonal(x, (s,s))
diagonal(x::AbstractArray, tup::Tuple{Symbol, Symbol}) = NamedDimsArray{tup}(Diagonal(x))

####################
