using .AxisKeys
using RecipesBase

@recipe function _fun(a::Union{KeyedArray{<:Any,1}, NamedDimsArray{<:Any,<:Any,1,<:KeyedArray}})
    label --> ""

    if hasnames(a)
        xaxis --> string(getnames(a,1))
    end

    axiskeys(a,1), nameless(AxisKeys.keyless(a))
end

@recipe function _fun(a::Union{KeyedArray{<:Any,2}, NamedDimsArray{<:Any,<:Any,2,<:KeyedArray}})

    if hasnames(a)
        xaxis --> string(getnames(a,1))

        ystub = string(getnames(a,2)) * " = "
    else
        ystub = ""
    end
    label --> permutedims(ystub .* string.(axiskeys(a,2)))

    # if hasmeta(a)
    #     yaxis --> string(meta(a))
    # end

    nameless(AxisKeys.keyless(a))
end
