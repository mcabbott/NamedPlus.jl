
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

export ⁺

"""
    NamedInt(a=3)

An `Int` with a name attached! So that `zeros(m,n) isa NamedDimsArray`, essentially.
`n'` will add a prime to the name.
`[f(i) for i in 1:n]` will inherit the name, because `1:n` does.
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

Base.show(io::IO, x::NamedInt{L}) where {L} = print(io, x.val, "⁺")
Base.show(io::IO, ::MIME"text/plain", x::NamedInt{L}) where {L} =
    print(io, "NamedInt($L=", x.val, ")")
const ⁺ = NamedInt(1, :_)
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

# for N in 0:10
#     pres = [Symbol(:pre_,n) for n=1:N]
#     ints = map(pre -> :($pre::Integer), pres)
#     for fun in [:zeros, :ones]

#         @eval Base.$fun(nis::NamedInt...) = $fun(Float64, nis...)
#         @eval Base.$fun($(ints...), ni::NamedInt, post::Integer...) =
#             $fun(Float64, $(pres...), ni, post...)

#         @eval function Base.$fun(T::Type, $(ints...), ni::NamedInt, post::Integer...)
#             arg = ($(pres...), ni, post...)
#             data = $fun(T, value.(arg)...)
#             NamedDimsArray(data, name.(arg))
#         end

#     end

#     @eval function Base.fill(val, $(ints...), ni::NamedInt, post::Integer...)
#         arg = ($(pres...), ni, post...)
#         data = fill(val, value.(arg)...)
#         NamedDimsArray(data, name.(arg))
#     end

#     i_or_colon = map(pre -> :($pre::Union{Integer,Colon}), pres)

#     @eval function Base.reshape(x::NamedDimsArray, $(i_or_colon...), ni::NamedInt, post::Union{Integer,Colon}...)
#         arg = ($(pres...), ni, post...)
#         data = reshape(nameless(x), value.(arg)...)
#         NamedDimsArray(data, name.(arg))
#     end

# end

Base.zeros(nsz::NamedInt...) = zeros(Float64, nsz...)
Base.zeros(T::Type, nsz::NamedInt...) = NamedDimsArray(zeros(T, value.(nsz)...), name.(nsz))

Base.ones(nsz::NamedInt...) = zeros(Float64, nsz...)
Base.ones(T::Type, nsz::NamedInt...) = NamedDimsArray(ones(T, value.(nsz)...), name.(nsz))

Base.fill(val, nsz::NamedInt...) = NamedDimsArray(fill(val, value.(nsz)...), name.(nsz))


####################
