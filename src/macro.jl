
#################### MEGA-MACRO ####################

export @named

using MacroTools

"""
    @named A{i,j} = rand(2,3)    # A ≈ NamedDimsArray(rand(2,3), (:i, :j))
    @named B = A{j,i}            # B = unname(A, (:j, :i))
    @named C{x,y} = A{j,i}       # C = NamedDimsArray(B, (:x, :y))

The `@named` macro can be used as a shorter way to add or remove names.
It abuses `{i,j}` notation because these labels are part of the type of `A`.
Note that the un-named `B` is sure to have the dimension that was `:j` as its first index.

    @named @tensor A[i,j] := B[i,k] * C[k,j]

The macro does two things to `@tensor` expressions:
The newly created output array `A` here is always a `NamedDimsArray{(:i,:k)}`.
And if `B` is a `NamedDimsArray`, then it must have names `(:i,:k)`, but any order is fine.

    @named D{i,j,y,x} = B .+ exp.(C)
    E = @named {i,j} = A .* B

The same macro can work on broadcasting, but it needs to be given the list of final names,
and cannot write into an existing array. If `C` is a `NamedDimsArray` then it will have
its labels permuted (lazily) to match `i,j,y,x` before broadcasting.
Ordinary arrays assumed to already match this order.

    D = @named softmax(A, dims={i})
    S = @named sum(f, A, dropdims={i})

Curly brackets here tell the macro to treat this like `sum(A, dims=:i)`, but this works on
functions which `NamedDims` hasn't extended.
Allows `fun(f, A; dims)` but not `fun(A; dims) do x ...`, nor other keywords.

    @named *′ = contract{k}      # *′(xs...) = Contract{(:k,)}(xs...)

For the special word `contract`, this defines a function as shown. Using a decorated
infix symbol (such as `*′` or `*ⱼ`) lets you call this as `B *′ C`.

    @named begin
        B{i,j} = rand(2,3)
        C{k,j} = rand(2,3)
        @tensor A[i,j] := B[i,k] * C[k,j]
        A{i,j,k} = B .+ exp.(C)
    end

The macro can be applied to blocks of code like this. Expressions without curly brackets
(or `@tensor`) will be treated as usual. Real type expressions may go horribly wrong.
"""
macro named(ex)
    esc(_named(ex, __module__))
end

_named(input_ex, mod) =
    MacroTools.postwalk(input_ex) do ex
        @capture(ex, @tensor lhs_ := rhs_) && return ex_tensor(rhs, lhs)
        @capture(ex, @tensor lhs_ = rhs_)  && return ex_tensor(ex)

        @capture(ex, @einsum ex) && error("@named doesn't yet work on @einsum")
        @capture(ex, @strided ex) && error("@named doesn't yet work on @strided")

        @capture(ex, lhs_{ind__} = rhs_) && return ex_cast(lhs, ind, rhs, mod)
        @capture(ex, {ind__} = rhs_)     && return ex_cast(gensym(:bc), ind, rhs, mod)

        @capture(ex, Z_{xyz__} = A_{ijk__}) && return ex_rename(Z, xyz, A, ijk)

        # Special words like contract must come before unname
        @capture(ex, g_ = contract{ijk__}) && return ex_fun(g, :Contract, ijk)

        if @capture(ex, fun_(args__, A_, dims={ijk__})) || @capture(ex, fun_(args__, A_; dims={ijk__}))
            return ex_dims(args, fun, A, ijk)
        elseif @capture(ex, fun_(args__, A_, dropdims={ijk__})) || @capture(ex, fun_(args__, A_; dropdims={ijk__}))
            return ex_dropdims(args, fun, A, ijk)
        end

        # @capture(ex, A_{ijk__} = B_) && return ex_addname(A, ijk, B) # clash!
        # @capture(ex, {ijk__} = B_)   && return ex_addname(gensym(:def), ijk, B)

        # This unname thing must come last
        if @capture(ex, A_ = B_{ijk__})
            B in (
                :NamedDimsArray, :(NamedDims.NamedDimsArray), :(NamedPlus.Contract),
                ) ||
            return ex_unname(A, B, ijk)
        end
        ex
    end

quotenodes(ijk::Vector) = map(quotenodes, ijk)
quotenodes(s::Symbol) = QuoteNode(s)
quotenodes(q::QuoteNode) = q

function ex_unname(A, B, ijk)
    stup = :( ($(quotenodes(ijk)...),) )
    :( $A = NamedDims.unname($B, $stup) )
end

function ex_addname(A, ijk, B)
    stup = :( ($(quotenodes(ijk)...),) )
    :( $A = NamedDims.NamedDimsArray($B, $stup) )
end

function ex_rename(Z, xyz, A, ijk)
    stup = :( ($(quotenodes(ijk)...),) )
    stup2 = :( ($(map(QuoteNode,xyz)...),) )
    :( $Z = NamedDims.NamedDimsArray(NamedDims.unname($A, $stup), $stup2) )
end

function ex_fun(f, s, ijk)
    @assert s == :Contract
    f == :* && @warn "are you sure you mean to define a new function `*`?"
    stup = :( ($(quotenodes(ijk)...),) )
    :( $f(x...) = NamedPlus.Contract{$stup}(x...) )
end

# This is an awful hack to avoid wrapping .+ with _permutenames
# Also note that simple assignments come here not to ex_addname(),
# so you will get NamedDimsArray(_permutenames(rand(2,3), ...), ...) which should be OK
function ex_cast(lhs, ind, rhs, mod)
    rhs isa Symbol && return ex_addname(lhs, ijk, rhs)

    tup = :( ($(map(QuoteNode, ind)...),) )
    # low = Meta.lower(mod, rhs)
    newright = MacroTools.postwalk(rhs) do x
        if x isa Symbol && !startswith(string(x),'.')
            # return :( NamedPlus._permutenames($x, $tup) )
            return :( NamedPlus.TransmuteDims.Transmute{$tup}($x) )
        end
        x
    end
    return :( $lhs = NamedDims.NamedDimsArray($newright, $tup) )
end

function ex_dims(args, fun, A, ijk)
    inds = map(QuoteNode, ijk)
    @gensym A′
    quote
        $A′ = $A # pulled out in case this is a calculation
        $fun($(args...), $A′; dims=NamedPlus._un_onetuple(NamedDims.dim($A′, ($(inds...),))) )
    end
end

function ex_dropdims(args, fun, A, ijk)
    inds = map(QuoteNode, ijk)
    @gensym A′ dims
    quote
        $A′ = $A # pulled out in case this is a calculation
        $dims = NamedPlus._un_onetuple(NamedDims.dim($A′, ($(inds...),)))
        dropdims($fun($(args...), $A′; dims=$dims); dims=$dims)
    end
end

_un_onetuple(x) = x
_un_onetuple(x::Tuple{<:Any}) = first(x)

function ex_tensor(ex, left=nothing)
    out = quote end
    tex = MacroTools.postwalk(ex) do x
        if @capture(x, A_[ijk__])
            Aname, Aperm = gensym(A), gensym(:perm)
            inds = map(QuoteNode, ijk)
            append!(out.args, (quote
                if $A isa NamedPlus.NamedUnion
                    $Aperm = dim($A, ($(inds...),))
                    $Aname = Base.permutedims(TensorOperations.Strided.maybestrided(NamedPlus.unname($A)), $Aperm)
                else
                    $Aname = $A
                end
            end).args)
            return :( $Aname[$(ijk...)] )
        end
        x
    end
    # case of ex = @tensor A[] = B[] * C[], in-place
    if left===nothing
        push!(out.args, tex) # returns unwrapped A, fix this... TODO
        return out

    # case of @tensor A[] := B[] * C[], new array created
    else
        @capture(left, A_[ijk__]) || error("can't understand @tensor LHS, $left")
        Aname = gensym(A)
        inds = map(QuoteNode, ijk)
        append!(out.args, (quote
            @tensor $Aname[$(ijk...)] := $tex
            NamedDims.NamedDimsArray{($(inds...),)}($Aname)
        end).args)
        return :( $A = $out )
    end
end


#################### DROPDIMS ####################

export @dropdims

using MacroTools

"""
    @dropdims sum(A; dims=1)

Macro which wraps such reductions in `dropdims(...; dims=1)`.
Allows `sum(A; dims=1) do x stuff end`,
and works on whole blocks of code like `@views`.
Does not handle other keywords, like `reduce(...; dims=..., init=...)`.
"""
macro dropdims(ex)
    esc(_dropdims(ex))
end

_dropdims(ex) =
    MacroTools.postwalk(ex) do x
        if @capture(x, red_(args__, dims=d_)) || @capture(x, red_(args__; dims=d_))
            :( dropdims($x; dims=$d) )
        elseif @capture(x, dropdims(red_(args__, dims=d1_); dims=d2_) do z_ body_ end) ||
               @capture(x, dropdims(red_(args__; dims=d1_); dims=d2_) do z_ body_ end)
            :( dropdims($red($z -> $body, $(args...); dims=$d1); dims=$d2) )
        else
            x
        end
    end

####################
