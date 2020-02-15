
#################### CONTRACT ####################

import .TensorOperations: contract!

# import NamedPlus: contract, _contract, contract∇A; using TensorOperations: contract!

struct One <: Integer end
Base.promote_rule(::Type{<:One}, ::Type{T}) where {T<:Number} = T
Base.convert(::Type{T}, ::One) where {T<:Number} = convert(T, 1)

"""
    contract(A, B, s)
    contract(A, B)

This is like `mul` except that it accepts not just matrices but also higer-dimensional arrays,
using `TensorOperations.contract!`.
By default it contracts over all names shared between `A` and `B`,
if there are none then it is equivalent to `outer(A,B)`.

    contract(γ, A, B)
    contract(A', B)    # not yet!

To multiply by a number `γ`, write it first. To conjugate all elements of one matrix,
write `adjoint(A)`, which thanks to some piracy is no longer an error.
"""
contract(A::NamedDimsArray, B::NamedDimsArray, sys::Symbol...) = contract(One(), A, B, Val(sys))

@generated function contract(A::NamedDimsArray{LA}, B::NamedDimsArray{LB}) where {LA,LB}
    s = filter(n -> Base.sym_in(n, LB), LA)
    :(contract(One(), A, B, Val($s)))
end

@generated function contract(γ::Number, A::NamedDimsArray{LA}, B::NamedDimsArray{LB}) where {LA,LB}
    s = filter(n -> Base.sym_in(n, LB), LA)
    :(contract(γ, A, B, Val($s)))
end

@generated function contract(γ::Number, A::NamedDimsArray{LA,TA,NA}, B::NamedDimsArray{LB,TB,NB}, ::Val{cs}) where {LA,LB,TA,TB,NA,NB,cs}
    cindA = NamedDims.dim(LA, cs)
    oindA = Tuple(setdiff(1:NA, cindA))
    cindB = NamedDims.dim(LB, cs)
    oindB = Tuple(setdiff(1:NB, cindB))
    for s in cs
        count(isequal(s), LA) == 1 || throw(ArgumentError("contraction index $s must appear exactly once in names(A) = $LA"))
        count(isequal(s), LB) == 1 || throw(ArgumentError("contraction index $s must appear exactly once in names(B) = $LB"))
    end

    indCinoAB = Tuple(vcat(1:length(oindA), length(oindA) .+ (1:length(oindB))))
    TC = Base.promote_type(TA,TB)

    @gensym C
    out = quote
        $C = _contract(γ, nameless(A), :N, nameless(B), :N, $oindA, $cindA, $oindB, $cindB, $indCinoAB)
    end

    LC = (map(d -> LA[d], oindA)..., map(d -> LB[d], oindB)...)
    if length(oindA) == length(oindB) == 0
        push!(out.args, :(first($C)))
    else
        push!(out.args, :(NamedDimsArray{$LC}($C)))
    end
    out
end


"""
    _contract(γ, A, :N, B, :N, ...)

This is just like `contract!` except that it makes C by itself,
and that it understands `γ::One` as a constant 1.
It does not handle names at all.
"""
function _contract(γ, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
    sizes = (map(d -> size(A,d), oindA)..., map(d -> size(B,d), oindB)...)
    sizeC = map(i -> sizes[i], indCinoAB) # this could be out by an invperm type thing
    TC = Base.promote_type(eltype(A), eltype(B))
    C = similar(A, TC, sizeC)
    α = γ isa One ? true : γ
    contract!(α, A, conjA, B, :N, false, C, oindA, cindA, oindB, cindB, indCinoAB)
end

# Hack to let you conjugate one array easily?
# for N in 3:10
#     @eval Base.adjoint(A::NamedDimsArray{L,T,$N}) where {L,T} = Adjoint(A)
# end

#=
using NamedPlus, TensorOperations

@named begin
    m{i,j} = rand(Int8, 3,4)
    g = [n^i for n in 1:20, i in 1:3]
end

@btime mul($m, $g) # 485.713 ns
@btime mul($m, $g, :i) # 490

@btime contract($m, $g) # 1.213 μs, was 4.270 μs before adding @generated stuff

@btime contract($m, $g, :i)
@btime (() -> contract($m, $g, :i))() # 5.244 μs
@btime contract($m, $g, Val((:i,)))   # 1.211 μs

cc(m,g) = contract!(true, parent(m), :N, parent(g), :N, false, similar(parent(g),4,20),
    (2,), (1,), (1,), (2,), (1,2))
ccn(m,g) = NamedDimsArray(
    contract!(true, parent(m), :N, parent(g), :N, false, similar(parent(g),4,20),
    (2,), (1,), (1,), (2,), (1,2)), (:j, :n))

@btime cc($m, $g) # 1.207 μs
@btime ccn($m, $g) # 1.964 μs

@code_warntype mul(m, g) # ok!
@code_warntype contract(m, g) # now ok!
@code_warntype cc(m, g)
@code_warntype ccn(m, g)

=#
# contract!(true, nameless(A), :N, nameless(B), :N, false, C, $oindA, $cindA, $oindB, $cindB, $indCinoAB)

#################### GRADIENT ####################

using TupleTools
using ZygoteRules: @adjoint

@adjoint (::Type{T})(x) where {T<:NamedDimsArray} = T(x), Δ -> (nameless(Δ),)

@adjoint nameless(x::NamedDimsArray{L}) where {L} = nameless(x), Δ -> (NamedDimsArray{L}(Δ),)

# Adapted from TensorTrack.jl

@adjoint function _contract(α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
    C = _contract(α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
    function back(Δ)
        ∇A = contract∇A(Δ, α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
        ∇B = contract∇B(Δ, α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)

        ∇α = α isa One ? nothing : contract∇α(Δ, α, C)

        (∇α, ∇A, nothing, ∇B, nothing, nothing, nothing, nothing, nothing, nothing)
    end
    return C, back
end

findint(n::Int, tup::Tuple)::Int = findfirst(==(n), tup)

function contract∇A(Δ, α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
    indAinoΔB = TupleTools.invperm((oindA..., cindA...))

    oindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))
    cindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))

    ∇A = _contract(α, Δ, conjA, B, conjB, oindΔ, cindΔ, cindB, oindB, indAinoΔB)
end

function contract∇B(Δ, α, A, conjA, B, conjB, oindA, cindA, oindB, cindB, indCinoAB)
    indBinoAΔ = TupleTools.invperm((cindB..., oindB...))

    oindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))
    cindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))

    ∇B = _contract(α, A, conjA, Δ, conjB, cindA, oindA, oindΔ, cindΔ, indBinoAΔ)
end

function contract∇α(Δ, α, C)
    allind = ntuple(identity, ndims(C))
    _contract(One(), Δ, :N, C, :N, allind, (), allind, (), ())
end

####################
