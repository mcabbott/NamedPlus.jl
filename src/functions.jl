
"""
    @through f(x::NamedUnion, y::Array, z::Int)

This defines a function which first unwraps `x`, tben applies `f`.
If the result is an `AbstractArray`, adds names `output_names(f, x, y, z)`.
"""
macro through(ex)
    @capture(ex, fun_(args__) || error("wtf")
    in = args # could replace Named -> NamedUnion etc?
    un = map(args) do a
        @capture(a, x_::type_) ?  :(nameless($x)) : a
    end
    sy = map(a -> @capture(a, x_::Type_) ? x : a, args)
    @gensym out
    quote
        function $fun($(in...))
            $out = $fun($(un...))
            if $out isa AbstractArray
                NamedDimsArray($out, output_names($f, $(sy...)))
            else
                $out
            end
        end
    end |> esc
end

# TODO make it handle keywords, including dims

output_names(::Function, A::NamedUnion; kw...) = dimnames(A)

output_names(::typeof(*), A, B) = NamedDims.matrix_prod_names(dimnames(A), dimnames(B))


@through Base.sum(x::NamedUnion; dims=:)
@through Base.sum(f, x::NamedUnion; dims=:)


@through Base.:*(x::NamedUnion, y::NamedUnion) # you'll need lots for ambiguity reasons!

