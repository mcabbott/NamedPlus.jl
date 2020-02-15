
using .OMEinsum: einsum, EinCode

"""
    contract(A, B, C, s)
    contract(A, B, C)

Star contraction, on either a specified name `s` shared by all arrays,
or else on all names shared. Needs `using OMEinsum`.
```
julia> using NamedPlus, OMEinsum

julia> A, B, C = ones(i=3, s=10), ones(s=10, j=5), ones(k=6, s=10);

julia> contract(A, B, C) |> summary
"3×5×6 named(::Array{Float64,3}, (:i, :j, :k))"

julia> contract(C, A, B) |> summary
"6×3×5 named(::Array{Float64,3}, (:k, :i, :j))"
```
"""
contract(A::NamedDimsArray, B::NamedDimsArray, C::NamedDimsArray) =
    _contract(A, B, C)
contract(A::NamedDimsArray, B::NamedDimsArray, C::NamedDimsArray, D::NamedDimsArray) =
    _contract(A, B, C, D)
# if you don't specify the summed index:
function _contract(ABC::NamedDimsArray...)
    cs = Tuple(intersect(getnames.(ABC)...))
    contract(Val(cs), ABC...)
end

# if you do specify:
contract(A::NamedDimsArray, B::NamedDimsArray, C::NamedDimsArray, cs::Symbol...) =
    contract(Val(cs), A,B,C)
contract(A::NamedDimsArray, B::NamedDimsArray, C::NamedDimsArray, D::NamedDimsArray, cs::Symbol...) =
    contract(Val(cs), A,B,C,D)

@generated function contract(::Val{cs}, ABC::NamedDimsArray...) where {cs}
    Ls = map(A -> A.parameters[1], ABC)
    Ly = Tuple(union(map(L -> filter(!in(cs), L), Ls)...))

    for L in Ls, c in cs
        count(isequal(c), L) == 1 || throw(ArgumentError("contraction index $c must appear exactly once in names(A) = $L"))
    end

    Lall = union(Ls...)
    Iy = map(n -> findfirst(isequal(n), Lall), Ly)
    Is = map(L -> map(n -> findfirst(isequal(n), Lall), L), Ls)
    quote
        NamedDimsArray{$Ly}(einsum(EinCode($Is, $Iy), nameless.(ABC)))
    end
end

"""
    const *ᵇ = batchmul(:b)
    C = batchmul(:b, A, B)
    C = A *ᵇ B

Batched matrix multiplication, ``C_ikb = A_ijb B_jkb`` and generalisations.
Needs `using OMEinsum`.
```
julia> using NamedPlus, OMEinsum

julia> A, B = ones(i=3, j=4, b=2), ones(j=4, k=5, b=2);

julia> const *ᵇ = batchmul(:b)
#217 (generic function with 1 method)

julia> C = A *ᵇ B;

julia> summary(C)
"3×5×2 named(::Array{Float64,3}, (:i, :k, :b))"

julia> C[:,:,1] == A[:,:,1] * B[:,:,1]
true
```
"""
batchmul(b::Symbol) = (A, B) -> batchmul(Val(b), A, B)

batchmul(b::Symbol, A::NamedDimsArray, B::NamedDimsArray) = batchmul(Val(b), A, B)

@generated function batchmul(::Val{b}, A::NamedDimsArray{LA}, B::NamedDimsArray{LB}) where {LA, LB, b}
    LA_b = filter(!isequal(b), LA)
    LB_b = filter(!isequal(b), LB)
    cs = Tuple(intersect(LA_b, LB_b))
    Ly = (filter(!in(cs), (LA_b..., LB_b...,))..., b)

    for L in (LA, LB)
        count(isequal(b), L) == 1 || throw(ArgumentError("batch index $c must appear exactly once in names(A) = $L"))
        for c in cs
            count(isequal(c), L) == 1 || throw(ArgumentError("contraction index $c must appear exactly once in names(A) = $L"))
        end
    end

    Lall = union(LA, LB)
    IA = map(n -> findfirst(isequal(n), Lall), LA)
    IB = map(n -> findfirst(isequal(n), Lall), LB)
    Iy = map(n -> findfirst(isequal(n), Lall), Ly)
    quote
        NamedDimsArray{$Ly}(einsum(EinCode(($IA, $IB), $Iy), (nameless(A), nameless(B))))
    end
end

