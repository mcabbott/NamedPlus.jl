
#################### CONTRACT ####################

import .TensorOperations: contract!

# import NamedPlus: contract; using TensorOperations: contract!

"""
    contract(A, B, s)
    contract(A, B)

This is like `mul` except that it accepts not just matrices but also higer-dimensional arrays.
If no names are provided, then it contracts all names in common between `A` and `B`.
This is just a wrapper for `TensorOperations.contract!`.

    contract(γ, A, B)
    contract(A', B)

To multiply by a number `γ`, write it first. To conjugate all elements of one matrix,
write `adjoint(A)`, which thanks to some piracy is no longer an error.
"""
contract(A::NamedDimsArray, B::NamedDimsArray, s::Symbol) = contract(A, B, Val((s,)))

@generated function contract(A::NamedDimsArray{LA}, B::NamedDimsArray{LB}) where {LA,LB}
    s = filter(n -> Base.sym_in(n, LB), LA)
    :(contract(A, B, Val($s)))
end


@generated function contract(γ::Number, A::NamedDimsArray{LA,TA,NA}, B::NamedDimsArray{LB,TB,NB}, ::Val{s}) where {LA,LB,TA,TB,NA,NB,s}
    cindA = NamedDims.dim(LA, s)
    oindA = Tuple(setdiff(1:NA, cindA))

    cindB = NamedDims.dim(LB, s)
    oindB = Tuple(setdiff(1:NB, cindB))

    indCinoAB = Tuple(vcat(1:length(oindA), length(oindA) .+ (1:length(oindB))))
    TC = Base.promote_type(TA,TB)
    # @show oindA cindA oindB cindB indCinoAB

    sizeC = (map(d -> :(size(A,$d)), oindA)..., map(d -> :(size(B,$d)), oindB)...)
    sizeCint = :(map(Int, ($(sizeC...),) )) # avoid NamedInt!
    out = quote
        C = similar(nameless(A), $TC, $sizeCint)
        contract!(true, nameless(A), :N, nameless(B), :N, false, C, $oindA, $cindA, $oindB, $cindB, $indCinoAB)
    end

    LC = (map(d -> LA[d], oindA)..., map(d -> LB[d], oindB)...)
    if length(sizeC) == 0
        push!(out.args, :(first(C)))
    else
        push!(out.args, :(NamedDimsArray{$LC}(C)))
    end
    out
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
# Same as in TensorTrack.jl

using ZygoteRules: @adjoint

@adjoint function contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms),
    Δ -> ∇contract(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
end

function ∇contract(Δ, α::Tα, A::TA, conjA, B::TB, conjB, β::Tβ, C::TC, oindA, cindA, oindB, cindB, indCinoAB, syms) where {Tα,TA,Tβ,TB,TC}

    ∇A = contract∇A(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    ∇B = contract∇B(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    ∇C = any∇C(Δ,β)

    ∇α = 0 # false # contract∇α(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB)
    ∇β = 0 # false # contract∇β(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB)

    return (∇α, ∇A, nothing, ∇B, nothing, ∇β, ∇C, nothing, nothing, nothing, nothing, nothing, nothing)
end


findint(n::Int, tup::Tuple)::Int = findfirst(i->i==n, tup)

function contract∇A(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms=nothing)

    indAinoΔB_old = ntuple(i->i, ndims(A))
    indAinoΔB = TupleTools.invperm((oindA..., cindA...))
    ∇VERBOSE && println("indAinoΔB_old = ",indAinoΔB_old, "  , indAinoΔB = ",indAinoΔB)
    oindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))
    cindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))

    simA = similar(A)
    # indA = ntuple(i->i, ndims(A))
    # simA = cached_similar_from_indices(sym_glue(syms, :_c∇A), eltype(A), indA, (), A, :N)

    ∇A = contract!(α, Δ, conjA, B, conjB, false, simA, oindΔ, cindΔ, cindB, oindB, indAinoΔB, sym_suffix(syms, :_∇A))
end

function contract∇B(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms=nothing)

    indBinoAΔ = TupleTools.invperm((cindB..., oindB...))
    oindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))
    cindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))

    simB = similar(B)
    # indB = ntuple(i->i, ndims(B))
    # simB = cached_similar_from_indices(sym_glue(syms, :_c∇B), eltype(B), indB, (), B, :N)

    ∇B = contract!(α, A, conjA, Δ, conjB, false, simB, cindA, oindA, oindΔ, cindΔ, indBinoAΔ, sym_suffix(syms, :_∇B))
end

sym_suffix(syms, suffix) = Symbol.(syms, suffix)
sym_suffix(::Nothing, suffix) = nothing

sym_glue(syms, suffix) = Symbol(syms..., suffix)
sym_glue(::Nothing, suffix) = Symbol(:Δnew, suffix)

####################
