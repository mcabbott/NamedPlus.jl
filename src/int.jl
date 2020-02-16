
#=
This is a terrible idea, but I started to wonder whether you could do this:

x = NamedDimsArray(rand(2,5), (:x, :y))
d,k = size(x)
M = zeros(d,d)
for r = 1:d
    M[r,1] = r
end
M * x
M2 = [i/j for i in 1:d', j in 1:d]
M2 * x

=#

#################### NAMED-INT TYPE ####################

export ᵅ

"""
    NamedInt(μ=3)

An `Int` with a name attached!
So that `zeros(m,n) isa NamedDimsArray`, essentially.
`n'` will add a prime to the name.
`[f(i) for i in 1:n]` will inherit the name, because `1:n` does.
Printed `3\\^alpha` in short; if you enter this you get `NamedInt(_=3)`.
"""
struct NamedInt{L} <: Integer
    val::Int
    NamedInt(;kw...) = new{first(kw.itr)}(first(kw.data))
    NamedInt(val::Integer, symb::Symbol) = new{symb}(val)
    NamedInt{symb}(val::Integer) where {symb} = new{symb}(val)
end
name(x::NamedInt{L}) where {L} = L
name(x::Number) = :_
value(x::NamedInt) = x.val
value(x::Number) = x

Base.show(io::IO, x::NamedInt{L}) where {L} = print(io, x.val, "ᵅ")
Base.show(io::IO, ::MIME"text/plain", x::NamedInt{L}) where {L} =
    print(io, "NamedInt($L = ", x.val, ")")

# These operation preserve names, firstly to make printout re-pastable, but also...
const ᵅ = NamedInt(1, :_)
Base.:*(x::Integer, y::NamedInt{L}) where {L} = NamedInt(x * y.val, L)
Base.:*(x::NamedInt{Lx}, y::NamedInt{Ly}) where {Lx, Ly} = NamedInt(x.val * y.val, _join(Lx, Ly))
# Base.literal_pow(::typeof(^), x::NamedInt{L}, ::Val{2}) where {L} =
#     NamedInt(Base.literal_pow(^, x.val, Val(2)), _join(L,L))

# These operations forget the name:
Base.literal_pow(::typeof(^), x::NamedInt, p::Val) = Base.literal_pow(^, x.val, p)
for f in [:abs, :abs2, :sign, :string, :Int, :float, :-, :OneTo]
    @eval Base.$f(x::NamedInt) = Base.$f(x.val)
end
for (T,S) in [(:NamedInt, :Integer), (:Integer, :NamedInt), (:NamedInt, :NamedInt)]
    for op in [:<, :>, :>=, :<=]
        @eval Base.$op(x::$T, y::$S) = $op(value(x), value(y))
    end
end
Base.convert(::Type{T}, x::NamedInt) where {T<:Number} = convert(T, x.val)
Base.promote_rule(::Type{<:NamedInt}, ::Type{T}) where {T<:Number} = T

function Base.:(==)(x::NamedInt{Lx}, y::NamedInt{Ly}) where {Lx, Ly}
    if isequal(x.val, y.val)
        Lx === :_ || Ly == :_ && return true
        Lx === Ly || @error "mismatched names on NamedInt($Lx=$(x.val)) & NamedInt($Ly=$(y.val))"
        true
    else
        false
    end
end

# from base/show.jl:2122
# dims2string(d) = isempty(d) ? "0-dimensional" :
#                  length(d) == 1 ? "$(d[1])-element" :
#                  join(map(string,d), '×')
Base.dims2string(d::Tuple{NamedInt}) = string(d[1].val, "-element")

nameless(x::NamedInt) = value(x)
nameless(xs::Tuple{NamedInt, Vararg{Integer}}) = map(value, xs)
nameless(xs::Tuple{Integer, NamedInt, Vararg{Integer}}) = map(value, xs)
nameless(xs::Tuple{NamedInt, NamedInt, Vararg{Integer}}) = map(value, xs)

#################### USING IT ####################

# Base.size(x::NamedDimsArray{L}) where {L} = map(NamedInt, size(parent(x)), L)
for N in 1:10 # hack to not exactly overwrite existing defn.
    @eval begin
        Base.size(x::NamedDimsArray{L,T,$N}) where {L,T} = map(NamedInt, size(parent(x)), L)
        Base.size(x::NamedDimsArray{L,T,$N}, d::Integer) where {L,T} =
            d <= $N  ? NamedInt(size(parent(x),d), L[d]) : 1
        # Base.size(x::NamedDimsArray{L,T,$N}, s::Symbol) where {L,T} = size(x, NamedDims.dim(x,s))
    end
end
Base.length(x::NamedDimsArray{L,T,1}) where {L,T} = NamedInt(length(parent(x)), L[1])

prime(x::NamedInt) = NamedInt(x.val, prime(name(x)))

LinearAlgebra.adjoint(x::NamedInt) = prime(x)

(c::Colon)(start::Integer, stop::NamedInt{L}) where {L} =
    NamedDimsArray(c(start, stop.val), L)

for fun in [:zeros, :ones, :rand, :randn]
    newfun = Symbol(:named_, fun)
    @eval begin
        Base.$fun(ni::NamedInt, rest::Integer...) = $newfun(Float64, ni, rest...)
        Base.$fun(i::Integer, ni::NamedInt, rest::Integer...) = $newfun(Float64, i, ni, rest...)
        Base.$fun(ni::NamedInt, ni′::NamedInt, rest::Integer...) = $newfun(Float64, ni, ni′, rest...)

        Base.$fun(T::Type, ni::NamedInt, rest::Integer...) = $newfun(T, ni, rest...)
        Base.$fun(T::Type, i::Integer, ni::NamedInt, rest::Integer...) = $newfun(T, i, ni, rest...)
        Base.$fun(T::Type, ni::NamedInt, ni′::NamedInt, rest::Integer...) = $newfun(T, ni, ni′, rest...)

        $newfun(T::Type, sz::Integer...) = NamedDimsArray($fun(T, value.(sz)...), name.(sz))
    end
end

Base.fill(val, ni::NamedInt, rest::Integer...) = named_fill(val, ni, rest...)
Base.fill(val, i::Integer, ni::NamedInt, rest::Integer...) = named_fill(val, i, ni, rest...)
Base.fill(val, ni::NamedInt, ni′::NamedInt, rest::Integer...) = named_fill(val, ni, ni′, rest...)

named_fill(val, sz::Integer...) = NamedDimsArray(fill(val, value.(sz)...), name.(sz))

Base.similar(A::AbstractArray, ni::NamedInt, rest::Integer...) = named_similar(A, eltype(A), ni, rest...)
Base.similar(A::AbstractArray, i::Integer, ni::NamedInt, rest::Integer...) = named_similar(A, eltype(A), i, ni, rest...)
Base.similar(A::AbstractArray, ni::NamedInt, ni′::NamedInt, rest::Integer...) = named_similar(A, eltype(A), ni, ni′, rest...)

Base.similar(A::AbstractArray, T::Type, ni::NamedInt, rest::Integer...) = named_similar(A, T, ni, rest...)
Base.similar(A::AbstractArray, T::Type, i::Integer, ni::NamedInt, rest::Integer...) = named_similar(A, T, i, ni, rest...)
Base.similar(A::AbstractArray, T::Type, ni::NamedInt, ni′::NamedInt, rest::Integer...) = named_similar(A, T, ni, ni′, rest...)

named_similar(A, T, sz::Integer...) = NamedDimsArray(similar(nameless(A), T, value.(sz)...), name.(sz))

(::Type{Array{T}})(::UndefInitializer, ni::NamedInt, rest::Integer...) where {T} = named_new(T, ni, rest...)
(::Type{Array{T}})(::UndefInitializer, i::Integer, ni::NamedInt, rest::Integer...) where {T} = named_new(T, i, ni, rest...)
(::Type{Array{T}})(::UndefInitializer, ni::NamedInt, ni′::NamedInt, rest::Integer...) where {T} = named_new(T, ni, ni′, rest...)
named_new(::Type{T}, sz::Integer...) where {T} = NamedDimsArray(Array{T}(undef, value.(sz)...), name.(sz))

#################### RESHAPE ####################

const CorI = Union{Colon,Integer}
value(::Colon) = (:)
name(::Colon) = :_

Base.reshape(A::AbstractArray, ni::NamedInt, rest::CorI...) = named_reshape(A, ni, rest...)
Base.reshape(A::AbstractArray, i::CorI, ni::NamedInt, rest::CorI...) = named_reshape(A, i, ni, rest...)
Base.reshape(A::AbstractArray, ni::NamedInt, ni′::NamedInt, rest::CorI...) = named_reshape(A, ni, ni′, rest...)

named_reshape(A::AbstractArray, sz::CorI...) = NamedDimsArray(reshape(unname(A), value.(sz)...), name.(sz))

Base.reshape(A::AbstractArray, sz::Tuple{NamedInt, Vararg{CorI}}) = named_reshape(A, sz...)
Base.reshape(A::AbstractArray, sz::Tuple{CorI, NamedInt, Vararg{CorI}}) = named_reshape(A, sz...)
Base.reshape(A::AbstractArray, sz::Tuple{NamedInt, NamedInt, Vararg{CorI}}) = named_reshape(A, sz...)

# function named_reshape(A::NamedUnion, sz::CorI...)
#     getnames(A)
#     target = name.(sz) # what do I want?
# end


####################
