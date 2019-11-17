using .AxisRanges
using RecipesBase

const RangeVec = Union{RangeArray{<:Any,1}, NamedDimsArray{<:Any,<:Any,1,<:RangeArray}}
const RangeMat = Union{RangeArray{<:Any,2}, NamedDimsArray{<:Any,<:Any,2,<:RangeArray}}

@recipe function _fun(a::RangeVec)
    label --> ""

    if hasnames(a)
        xaxis --> string(names(a,1))
    end

    ranges(a,1), parent(a)
end

@recipe function _fun(a::RangeMat)

    if hasnames(a)
        xaxis --> string(names(a,1))

        ystub = string(names(a,2)) * " = "
    else
        ystub = ""
    end
    label --> permutedims(ystub .* string.(ranges(a,2)))

    # if hasmeta(a)
    #     yaxis --> string(meta(a))
    # end

    ranges(a,1), parent(a)
end
