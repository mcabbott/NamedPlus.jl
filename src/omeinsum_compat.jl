# import NamedPlus: Contract, NamedUnion

using TupleTools

# @macroexpand @ein A[x,z] := xy[x,y] * yz[y,z]
# OMEinsum.einsum(OMEinsum.EinCode{((1, 2), (2, 3)), (1, 3)}(), (xy, yz))

# @macroexpand @ein A[i,j,k,a,b,c] := A[i,j,z,k] * B[a,z,b,c]
# :(A = OMEinsum.einsum(OMEinsum.EinCode{((1, 2, 3, 4), (5, 3, 6, 7)), (1, 2, 4, 5, 6, 7)}(), (A, B)))

#################### CONTRACT ####################

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

#################### OVERLOAD ####################

# This doesn't know names used in @ein
# So perhaps you should write an @named case for this too,
# which saves these for use by a compile-time function...
# Perhaps you could do that to @einsum too, pushing macro to compile-time


@generated function OMEinsum.einsum(code::OMEinsum.EinCode{xlists,ylist}, args::NTuple{N,NamedUnion}) where {xlists, ylist, N}
    length(xlists) == N || error("expected $(length(ixs)) arguments but got $N, for eincode = $code")
    xnames = map(NamedDims.names, args.parameters)
    idict = Dict()
    for x in 1:N
        for (ix, nx) in zip(xlists[x], xnames[x])
            nx == :_ && continue
            get!(idict, ix, nx) == nx  || error("mismatched names")
        end
    end
    outnames = Any[ QuoteNode(idict[iy]) for iy in ylist ]

    :( NamedDimsArray{($(outnames...),)}(
        OMEinsum.einsum(code, map(NamedDims.unname, args))
    ) )
end


# TODO make this accept anything with at least one name?
# @generated new name, @eval OMEinsum.einsum with Tuple{Any,Any, WeightedUnion, Any...}

####################
