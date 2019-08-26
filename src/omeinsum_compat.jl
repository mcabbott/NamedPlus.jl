# import NamedPlus: Contract, NamedUnion

using TupleTools

# @macroexpand @ein A[x,z] := xy[x,y] * yz[y,z]
# OMEinsum.einsum(OMEinsum.EinCode{((1, 2), (2, 3)), (1, 3)}(), (xy, yz))

# @macroexpand @ein A[i,j,k,a,b,c] := A[i,j,z,k] * B[a,z,b,c]
# :(A = OMEinsum.einsum(OMEinsum.EinCode{((1, 2, 3, 4), (5, 3, 6, 7)), (1, 2, 4, 5, 6, 7)}(), (A, B)))

#=
function Contract{dims}(xraws::NamedUnion...) where {dims}
    xs = map(canonise, xraws)

    all(map(x -> issubset(dims, NamedDims.names(x)), xs)) || error("contraction indices $dims must appear in every factor")
    allnames = TupleTools.vcat(map(NamedDims.names, xs)...)

    outnames = tuple_filter(i -> !(i in dims), tuple_unique(allnames))
    outnumbers = map(i -> findfirst(isequal(i), allnames), outnames)

    innumbers = map(x ->
        map(i -> findfirst(isequal(i), allnames), NamedDims.names(x)),
        xs)

    resparent = OMEinsum.einsum(OMEinsum.EinCode{innumbers, outnumbers}(), map(parent, xs))
    NamedDimsArray{outnames}(resparent)
end
=#

@generated function Contract{dims}(xs::NamedUnion...) where {dims}
    # xs = map(canon_names, xs) # unfinished

    all(map(x -> issubset(dims, NamedDims.names(x)), xs)) || error("contraction indices $dims must appear in every factor")
    allnames = TupleTools.vcat(map(NamedDims.names, xs)...)

    outnames = tuple_filter(i -> !(i in dims), tuple_unique(allnames))
    outnumbers = map(i -> findfirst(isequal(i), allnames), outnames)

    innumbers = map(x ->
        map(i -> findfirst(isequal(i), allnames), NamedDims.names(x)),
        xs)

    quote
        resparent = OMEinsum.einsum(OMEinsum.EinCode{$innumbers, $outnumbers}(), map(Base.parent, xs))
        NamedDimsArray{$outnames}(resparent)
    end
end

tuple_filter(f, t::Tuple) = _filter(f, t, ())
@inline function _filter(f, t::Tuple, r::Tuple)
    if f(first(t))
        return _filter(f, Base.tail(t), (r..., first(t)))
    else
        return _filter(f, Base.tail(t), r)
    end
end
_filter(f, t::Tuple{}, r::Tuple) = r

tuple_unique(t::Tuple) = _unique(t, ())
@inline function _unique(t::Tuple, r::Tuple)
    if first(t) in r
        return _unique(Base.tail(t), r)
    else
        return _unique(Base.tail(t), (r..., first(t)))
    end
end
_unique(t::Tuple{}, r::Tuple) = r
