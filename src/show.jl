
function Base.summary(io::IO, A::NamedDimsArray)
    print(io, Base.dims2string(size(A)), " NamedDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", getnames(A), ")")
end

function Base.print_matrix(io::IO, A::NamedDimsArray)
    s1 = string("↓ :", getnames(A,1)) * "  "
    # s1c = "\e[35m$(s1)\e[0m"
    if ndims(A)==2
        s2 = string(" "^length(s1), "→ :", getnames(A,2), "\n")
        # printstyled(io, s2, color=:magenta)
        # s2c = "\e[35m$(s2)\e[0m"
        print(io, s2)
    end
    ioc = IOContext(io, :displaysize => displaysize(io) .- (2, 0))
    Base.print_matrix(ioc, parent(A), s1) # s1c messes up the indent!
end

