
#################### SPLIT / COMBINE ####################

"""
    join(A, (:i, :j) => :i_j)
    join(A, :i, :j) # => :i_j by default

This replaces two dimensions `i, j` with a combined one `i_j`.

If the dimensions are adjecent and in the correct order
(and `A` is not a lazy `Transpose` etc) then this will be done by reshaping.
But if the dimensions aren't adjacent, or are in the wrong order,
then it will copy `A`.
"""
Base.join(A::NamedUnion, i::Symbol, j::Symbol) = join(A, (i,j) => _join(i,j))
Base.join(A::NamedUnion, ij::Tuple) = join(A, ij...)

function Base.join(A::NamedUnion, p::Pair{<:Tuple,Symbol})
    d1, d2 = NamedDims.dim(A, p.first)

    if d2 == d1+1 && nameless(A) isa StridedArray
        sz = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? size(A,d) :
            d==min(d1,d2) ? size(A,d1) * size(A,d2) :
            size(A, d+1)
        end
        nm = ntuple(ndims(A)-1) do d
            d < min(d1,d2) ? getnames(A,d) :
            d==min(d1,d2) ? p.second :
            getnames(A, d+1)
        end
        # @info "noperm"
        return NamedDimsArray{nm}(reshape(nameless(A), sz))

    elseif d2 == d1 + 1
        return join(copy(A), p)

    elseif d1 < d2
        # 1 2 3* 4 5 6** 7 8
        # 1 2 4 5 3* 6** 7 8
        perm = ntuple(ndims(A)) do d
            d < d1 ? d :
            d < d2-1 ? d+1 :
            d == d2-1 ? d1 :
            d == d2 ? d2 :
            d
        end
        # @info "d1 < d2" perm
        return join(permutedims(A, perm), p)
    elseif d1 > d2
        # 1 2 3** 4 5 6* 7 8
        # 1 2 4 5 6* 3** 7 8
        perm = ntuple(ndims(A)) do d
            d < d2 ? d :
            d < d1-1 ? d+1 :
            d == d1-1 ? d1 :
            d == d1 ? d2 :
            d
        end
        # @info "d1 > d2" perm
        return join(permutedims(A, perm), p)
    end
    error("not yet")
end

# @btime (() -> _join(:i, :j))() # 0 allocations
_join(i::Symbol, j::Symbol) = _join(Val(i), Val(j))
@generated _join(::Val{i}, ::Val{j}) where {i,j} = QuoteNode(Symbol(i, :_, j))

# @btime (() -> _split(Symbol("i⊗j")))()  # 0 allocations, but 4 μs!
# @btime (() -> _split($(QuoteNode(Symbol("i⊗j")))))()  # 0 allocations, but 4 μs!
# @btime (() -> _split(_join(:i, :j)))()  # 0 allocations, 1.4 ns
_split(ij::Symbol) = _split(Val(ij))
@generated _split(::Val{ij}) where {ij} = Tuple(map(QuoteNode∘Symbol, split(string(ij), '_')))

"""
    split(A, :i_j => (i=2, j=3))
    split(A, :i_j => (:i, :j), (2,3))

This replaces the dimension named `i_j` with separate `i, j` by reshaping.
The final size of the two new dimension should be given afterwards;
you may write `(2,:)` etc.

    split(A, :i_j => (:i, :j), B)

The final sizes can also bew read from another `B::NamedDimsArray` with names `i,j`.
"""
Base.split(A::NamedUnion, pair::Pair{Symbol,<:NamedTuple}) =
    split(A, pair.first => keys(pair.second), Tuple(pair.second))

function Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, sizes::Tuple)
    d0 = dim(A, pair.first)
    sz = ntuple(ndims(A)+1) do d
        d < d0 ? size(A,d) :
        d==d0 ? sizes[1] :
        d==d0+1 ? sizes[2] :
        size(A, d-1)
    end
    nm = ntuple(ndims(A)+1) do d
        d < d0 ? getnames(A,d) :
        d==d0 ? pair.second[1] :
        d==d0+1 ? pair.second[2] :
        getnames(A, d-1)
    end
    NamedDimsArray{nm}(reshape(nameless(A), sz))
end

Base.split(A::NamedUnion, pair::Pair{Symbol,<:Tuple}, B::NamedUnion) =
    split(A, pair, dim(B, pair.second))


#=
ABC = NamedDimsArray(rand(1:10, 2,3,4), (:a,:b,:c))

@btime (ABC -> join(ABC, :a, :b))($ABC)        # 9 μs
@code_warntype (ABC -> join(ABC, :a, :b))(ABC) # ::Any

@btime (ABC -> split(ABC, :c => (:c1, :c2), (2,2)))($ABC)        # 650 ns
@code_warntype (ABC -> split(ABC, :c => (:c1, :c2), (2,2)))(ABC) # ::NamedDimsArray{_A,Int64,_B,_C}
=#

####################
