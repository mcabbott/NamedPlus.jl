# NamedPlus.jl

This package exists to experiment with the arrays provided by 
[NamedDims.jl](https://github.com/invenia/NamedDims.jl). 

Things which work:
```julia
@named begin
    m{i,j} = rand(2,3)                 # matrix whose type has (:i,:j)
    g = [n^i for n in 1:12, i in 1:2]  # read generator's variables
end

t = split(g, :n => (j=3, k=4))   # just reshape,
join(t, (:i, :k) => :χ)          # copy if non-adjacent
ones(z=2) .+ rename(m, :i => :z) # useful rename

d,_ = size(m)     # NamedInt, which exists only for...
z = zeros(Int, d,d')
z .+ [sqrt(i) for i in 1:d, i′ in 1:2⁺]
```
