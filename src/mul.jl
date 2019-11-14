
#################### MATRIX MULTIPLICATION ####################

"""
    mul(A, B, s)
    mul(A, B) = A *ᵃ B

Matrix multiplication for two `NamedDimsArray`s, which automatically transposes as required.
If given a name `s`, it arranges to sum over this index.
If not, it looks for a name shared between the two. The infix form is typed `*\\^a<tab>`.
"""
function mul(A::NamedUnion, B::NamedUnion)
    namesA, namesB = getnames(A), getnames(B)
    found = map(namesA) do n
        Base.sym_in(n, namesB) ? n : nothing
    end
    ok = filter(!isnothing, found)
    isempty(ok) && error("no name in common between $namesA and $namesB")
    length(ok) > 1 && error("no unique way to contract $namesA and $namesB")
    # allunique(namesB) || error("repeated names in $namesB") # these will be caught later
    mul(A, B, first(ok))
end

const *ᵃ = mul

function mul(x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,1}, s::Symbol) where {Lx,Tx,Ly,Ty}
    s == Lx[1] == Ly[1] || contract_error(x, y, s)
    if Tx <: Number
        transpose(x) * y
    else
        first(permutedims(x) * y)
    end
end

function mul(x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,1}, s::Symbol) where {Lx,Tx,Ly,Ty}
    Lx[1] == Lx[2] && contract_error(x, y, s)
    if s == Lx[2] == Ly[1]
        x * y
    elseif s == Lx[1] == Ly[1]
        transpose1(x) * y
    else
        contract_error(x, y, s)
    end
end

function mul(x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,2}, s::Symbol) where {Lx,Tx,Ly,Ty}
    Ly[1] == Ly[2] && contract_error(x, y, s)
    if s == Lx[1] == Ly[1]
        transpose1(transpose1(x) * y)
    elseif s == Lx[1] == Ly[2]
        transpose1(transpose1(x) * transpose1(y))
    else
        contract_error(x, y, s)
    end
end

function mul(x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,2}, s::Symbol) where {Lx,Tx,Ly,Ty}
    Lx[1] == Lx[2] && contract_error(x, y, s)
    Ly[1] == Ly[2] && contract_error(x, y, s)
    if s == Lx[2] == Ly[1]
        x * y
    elseif s == Lx[1] == Ly[1]
        transpose1(x) * y
    elseif s == Lx[2] == Ly[2]
        x * transpose1(y)
    elseif s == Lx[1] == Ly[2]
        transpose1(x) * transpose1(y)
    else
        contract_error(x, y, s)
    end
end

function contract_error(x, y, s)
    msg = "cannot contract index :$s between arrays with indices $(getnames(x)) and $(getnames(x))"
    throw(DimensionMismatch(msg))
end

transpose1(x::AbstractArray{<:Number}) = transpose(x)
transpose1(x::AbstractArray) = permutedims(x)

####################
