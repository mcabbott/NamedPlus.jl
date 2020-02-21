
#################### MEGA-MACRO ####################

using MacroTools

"""
    @named A{i,j} = rand(2,3)    # A ≈ named(rand(2,3), (:i, :j))
    @named B = A{j,i}            # B = nameless(A, (:j, :i))
    @named C{x,y} = A{j,i}       # C = named(B, (:x, :y))

The `@named` macro can be used as a shorter way to add or remove names.
It abuses `{i,j}` notation because these labels are part of the type of `A`.
Note that the un-named `B` is sure to have the dimension that was `:j` as its first index.

    @named D{i,j,y,x} = B .+ exp.(C)
    E = @named {i,j} = A .* B

The same macro can work on broadcasting, but it needs to be given the list of final names.
If `C` is a `NamedDimsArray` then it will have
its labels permuted (lazily) to match `i,j,y,x` before broadcasting.
(Ordinary arrays at left alone, thus assumed to already match this order.)

    @named F = [f(x) for x in 1:10]      # NamedDimsArray([f(x) for ...], :x)
    G = @named [g(x) for y in 10:2:20]   # wrapdims([g(x) for ...], y=10:2:20)

Acting on comprehensions, it will use the variable name as a dimension name.
If the range is not simply `1:N`, and AxisRanges.jl is loaded,
then it will make a `RangeArray{...,NamedDimsArray{(:y,),...}}`.

    @named @tensor A[i,j] := B[i,k] * C[k,j]

The macro does two things to `@tensor` expressions (`using TensorOperations`).
The newly created output array `A` here is always a `NamedDimsArray{(:i,:k)}`.
And if `B` is a `NamedDimsArray`, then it must have names `(:i,:k)`, but any order is fine.

    @named begin
        B{i,j} = rand(2,3)
        C{k,j} = rand(4,3)
        @tensor A[i,j] := B[i,k] * C[k,j]
        A{i,j,k} .= B .+ exp.(C)
    end

The macro can be applied to blocks of code like this. Expressions without curly brackets
(or `@tensor`, or comprehensions) will be treated as usual.
Real type expressions may go horribly wrong.
"""
macro named(ex)
    esc(_named(ex, __module__))
end

_named(input_ex, mod) =
    MacroTools.postwalk(input_ex) do ex
        @capture(ex, @tensor lhs_ := rhs_) && return ex_tensor(rhs, lhs)
        @capture(ex, @tensor lhs_ = rhs_)  && return ex_tensor(ex)

        @capture(ex, @einsum ex) && error("@named doesn't yet work on @einsum")
        @capture(ex, @ein ex) && error("@named doesn't yet work on OMEinsum.@ein")
        @capture(ex, @strided ex) && error("@named doesn't yet work on @strided")

        @capture(ex, Z_{xyz__} = A_{ijk__}) && return ex_rename(Z, xyz, A, ijk)

        # @capture(ex, A_{ijk__} = B_) && return ex_addname(A, ijk, B) # handled by ex_cast
        @capture(ex, lhs_{ind__} = rhs_) && return ex_cast(lhs, ind, rhs, mod)
        @capture(ex, {ind__} = rhs_)     && return ex_cast(gensym(:bc), ind, rhs, mod)
        @capture(ex, lhs_{ind__} .= rhs_) && return ex_incast(lhs, ind, rhs, mod)

        @capture(ex, [val_ for ind_ in ran_] ) &&
            return ex_comprehension(ex, val, ind, ran, mod)
        @capture(ex, [val_ for ind1_ in ran1_, ind2_ in ran2_] ) &&
            return ex_comprehension(ex, val, ind1, ran1, ind2, ran2, mod)

        # This nameless thing must come last
        if @capture(ex, A_ = B_{ijk__})
            B in (
                :NamedDimsArray, :(NamedDims.NamedDimsArray), :(NamedPlus.Contract),
                ) ||
            return ex_nameless(A, B, ijk)
        end
        ex
    end

quotenodes(ijk::Vector) = Expr(:tuple, map(quotenodes, ijk)...)
quotenodes(s::Symbol) = s == :(..) ? :(NamedPlus.EllipsisNotation.Ellipsis()) : QuoteNode(s)
quotenodes(ex::Expr) = @capture(ex, n_') ? QuoteNode(Symbol(n,'′')) : error("did not expect expression $ex")
quotenodes(q::QuoteNode) = q

function ex_nameless(A, B, ijk)
    tup = quotenodes(ijk)
    :( $A = NamedPlus.nameless($B, $tup) )
end

function ex_addname(A, ijk, B)
    tup = quotenodes(ijk)
    :( $A = NamedPlus.named($B, $tup) )
end

function ex_rename(Z, xyz, A, ijk)
    tup = quotenodes(ijk)
    tup_out = quotenodes(xyz)
    :( $Z = NamedPlus.named(NamedPlus.nameless($A, $tup), $tup_out) )
end

##### Comprehensions #####

function ex_comprehension(ex, val, ind, ran, mod) # [val_ for ind_ in ran_]
    name = QuoteNode(ind)
    out = :( NamedDims.NamedDimsArray{($name,)}($ex) )

    if (@capture(ran, start_:stop_) && start != 1) || @capture(ran, start_:step_:stop_)
        return quote
            isdefined($mod, :AxisRanges) ? RangeArray($out, ($ran,)) : $out
        end
    else
        return out
    end
end

function ex_comprehension(ex, val, ind1, ran1, ind2, ran2, mod)
    name1 = QuoteNode(ind1)
    name2 = QuoteNode(ind2)
    @gensym A
    out = :( NamedDims.NamedDimsArray{($name1,$name2)}($A) )

    needranges = false
    if (@capture(ran1, start_:stop_) && start != 1) || @capture(ran1, start_:step_:stop_)
        range1 = ran1
        needranges = true
    else
        range1 = :(axes($A, 1))
    end

    if (@capture(ran2, start_:stop_) && start != 1) || @capture(ran2, start_:step_:stop_)
        range2 = ran2
        needranges = true
    else
        range2 = :(axes($A, 2))
    end

    if needranges
        return quote
            $A = $ex
            isdefined($mod, :AxisRanges) ? RangeArray($out, ($range1,$range2)) : $out
        end
    else
        return quote
            $A = $ex
            $out
        end
    end
end

##### Broadcasting #####

function ex_cast(lhs, ind, rhs, mod)
    rhs isa Symbol && return ex_addname(lhs, ind, rhs)
    tup = quotenodes(ind)
    # This is an awful hack to avoid wrapping .+ with Transmute{},
    # everything else gets wrapped but Transmute{...}(sin) == sin
    newright = MacroTools.postwalk(rhs) do x
        if x isa Symbol && !startswith(string(x),'.')
            return :( NamedPlus.TransmuteDims.Transmute{$tup}($x) )
        end
        x
    end
    return :( $lhs = NamedPlus.named($newright, $tup) )
end

function ex_incast(lhs, ind, rhs, mod)
    tup = quotenodes(ind)
    newright = MacroTools.postwalk(rhs) do x
        if x isa Symbol && !startswith(string(x),'.')
            return :( NamedPlus.TransmuteDims.Transmute{$tup}($x) )
        end
        x
    end
    return :( NamedDims.NamedDimsArray($lhs, $tup) .= $newright )
end

##### Contractions @ @tensor #####

function ex_fun(f, s, ijk)
    @assert s == :Contract
    f == :* && @warn "are you sure you mean to define a new function `*`?"
    tup = :( ($(quotenodes(ijk)...),) )
    :( $f(x...) = NamedPlus.Contract{$tup}(x...) )
end

function ex_tensor(ex, left=nothing)
    out = quote end
    tex = MacroTools.postwalk(ex) do x
        if @capture(x, A_[ijk__])
            Aname, Aperm = gensym(A), gensym(:perm)
            inds = map(QuoteNode, ijk)
            append!(out.args, (quote
                # if $A isa NamedPlus.NamedUnion
                if $A isa NamedDims.NamedDimsArray
                    $Aperm = NamedDims.dim($A, ($(inds...),))
                    $Aname = Base.permutedims(TensorOperations.Strided.maybestrided(NamedPlus.nameless($A)), $Aperm)
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

####################
