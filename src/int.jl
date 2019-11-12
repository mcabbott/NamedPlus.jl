
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
end
name(x::NamedInt{L}) where {L} = L
name(x::Int) = :_
value(x::NamedInt) = x.val
value(x::Int) = x

Base.show(io::IO, x::NamedInt{L}) where {L} = print(io, x.val, "ᵅ")
Base.show(io::IO, ::MIME"text/plain", x::NamedInt{L}) where {L} =
    print(io, "NamedInt($L=", x.val, ")")
const ᵅ = NamedInt(1, :_)
Base.:*(x::Integer, y::NamedInt{L}) where {L} = NamedInt(x * y.val, L)
Base.:*(x::NamedInt{Lx}, y::NamedInt{Ly}) where {Lx, Ly} = NamedInt(x.val * y.val, Symbol(Lx, :_, Ly))

Base.convert(::Type{T}, x::NamedInt) where {T<:Number} = convert(T, x.val)
for f in [:abs, :abs2, :sign, :string, :Int, :float, :-]
    @eval Base.$f(x::NamedInt) = $f(x.val)
end
Base.promote_rule(::Type{<:NamedInt}, ::Type{T}) where {T<:Number} = T

(c::Colon)(start::Integer, stop::NamedInt{L}) where {L} =
    NamedDimsArray(c(start, stop.val), L)

#################### USING IT ####################

# Base.size(x::NamedDimsArray{L}) where {L} = map(NamedInt, size(parent(x)), L)
for N in 1:10 # hack to not exactly overwrite existing defn.
    @eval begin
        Base.size(x::NamedDimsArray{L,T,$N}) where {L,T} = map(NamedInt, size(parent(x)), L)
        Base.size(x::NamedDimsArray{L,T,$N}, d::Int) where {L,T} = NamedInt(size(parent(x),d), L[d])
        # Base.size(x::NamedDimsArray{L,T,$N}, s::Symbol) where {L,T} = size(x, NamedDims.dim(x,s))
    end
end

prime(x::NamedInt) = NamedInt(x.val, prime(name(x)))

LinearAlgebra.adjoint(x::NamedInt) = prime(x)

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
named_fill(val, sz::Integer...) = NamedDimsArray(fill(val, value.(sz)...), name.(sz))

Base.reshape(A::AbstractArray, ni::NamedInt, rest::Integer...) = named_reshape(A, ni, rest...)
Base.reshape(A::AbstractArray, i::Integer, ni::NamedInt, rest::Integer...) = named_reshape(A, i, ni, rest...)
Base.reshape(A::AbstractArray, ni::NamedInt, ni′::NamedInt, rest::Integer...) = named_reshape(A, ni, ni′, rest...)
named_reshape(A::AbstractArray, sz::Integer...) = NamedDimsArray(reshape(A, value.(sz)...), name.(sz))




####################
