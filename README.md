# NamedPlus.jl

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 

Things which work, they have docstrings with more info:
```julia
@named begin
    m{i,j} = rand(2,3)                 # matrix whose type has (:i,:j)
    g = [n^i for n in 1:12, i in 1:2]  # read generator's variables
end

t = split(g, :n => (j=3, k=4))   # just reshape,         could allow split(g, n = (j=3, k=4))
join(t, (:k, :i) => :χ)          # copy if non-adjacent  could allow join(t, χ = (:k, :i))? 
ones(z=2) .+ rename(m, :i => :z) # useful rename         could allow rename(m, i=:z)? z=i?
```
