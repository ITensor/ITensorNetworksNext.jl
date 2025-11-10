using NamedDimsArrays: dename
using TermInterface: iscall, maketerm, operation, sorted_arguments
using WrappedUnions: unwrap

lazy(x) = error("Not defined.")

# Walk the expression `ex`, modifying the
# operations by `opmap` and the arguments by `argmap`.
function walk(opmap, argmap, ex)
    if !iscall(ex)
        return argmap(ex)
    else
        return mapfoldl((args...) -> walk(opmap, argmap, args...), opmap(operation(ex)), arguments(ex))
    end
end
# Walk the expression `ex`, modifying the
# operations by `opmap`.
opwalk(opmap, a) = walk(opmap, identity, a)
# Walk the expression `ex`, modifying the
# arguments by `argmap`.
argwalk(argmap, a) = walk(identity, argmap, a)

# Generic lazy functionality.
function maketerm_lazy(type::Type, head, args, metadata)
    if head ≡ *
        return type(maketerm(Mul, head, args, metadata))
    else
        return error("Only mul supported right now.")
    end
end
function getindex_lazy(a::AbstractArray, I...)
    u = unwrap(a)
    if !iscall(u)
        return u[I...]
    else
        return error("Indexing into expression not supported.")
    end
end
function arguments_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments.")
    elseif ismul(u)
        return arguments(u)
    else
        return error("Variant not supported.")
    end
end
function children_lazy(a)
    return arguments(a)
end
function head_lazy(a)
    return operation(a)
end
function iscall_lazy(a)
    return iscall(unwrap(a))
end
function isexpr_lazy(a)
    return iscall(a)
end
function operation_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return error("No operation.")
    elseif ismul(u)
        return operation(u)
    else
        return error("Variant not supported.")
    end
end
function sorted_arguments_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments.")
    elseif ismul(u)
        return sorted_arguments(u)
    else
        return error("Variant not supported.")
    end
end
function sorted_children_lazy(a)
    return sorted_arguments(a)
end
ismul_lazy(a) = ismul(unwrap(a))
function abstracttrees_children_lazy(a)
    if !iscall(a)
        return ()
    else
        return arguments(a)
    end
end
function nodevalue_lazy(a)
    if !iscall(a)
        return unwrap(a)
    else
        return operation(a)
    end
end
materialize_lazy(a) = argwalk(unwrap, a)
using Base.Broadcast: materialize
copy_lazy(a) = materialize(a)
function equals_lazy(a1, a2)
    u1, u2 = unwrap.((a1, a2))
    if !iscall(u1) && !iscall(u2)
        return u1 == u2
    elseif ismul(u1) && ismul(u2)
        return arguments(u1) == arguments(u2)
    else
        return false
    end
end
function isequal_lazy(a1, a2)
    u1, u2 = unwrap.((a1, a2))
    if !iscall(u1) && !iscall(u2)
        return isequal(u1, u2)
    elseif ismul(u1) && ismul(u2)
        return isequal(arguments(u1), arguments(u2))
    else
        return false
    end
end
function hash_lazy(a, h::UInt64)
    h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
    # Use `_hash`, which defines a custom hash for NamedDimsArray.
    return _hash(unwrap(a), h)
end
function map_arguments_lazy(f, a)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments to map.")
    elseif ismul(u)
        return lazy(map_arguments(f, u))
    else
        return error("Variant not supported.")
    end
end
function substitute_lazy(a, substitutions::AbstractDict)
    haskey(substitutions, a) && return substitutions[a]
    !iscall(a) && return a
    return map_arguments(arg -> substitute(arg, substitutions), a)
end
function substitute_lazy(a, substitutions)
    return substitute(a, Dict(substitutions))
end
function printnode_lazy(io, a)
    # Use `printnode_nameddims` to avoid type piracy,
    # since it overloads on `AbstractNamedDimsArray`.
    return printnode_nameddims(io, unwrap(a))
end
function show_lazy(io::IO, a)
    if !iscall(a)
        return show(io, unwrap(a))
    else
        return AbstractTrees.printnode(io, a)
    end
end
function show_lazy(io::IO, mime::MIME"text/plain", a)
    summary(io, a)
    println(io, ":")
    !iscall(a) ? show(io, mime, unwrap(a)) : show(io, a)
    return nothing
end
add_lazy(a1, a2) = error("Not implemented.")
sub_lazy(a) = error("Not implemented.")
sub_lazy(a1, a2) = error("Not implemented.")
function mul_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return lazy(Mul([a]))
    elseif ismul(u)
        return a
    else
        return error("Variant not supported.")
    end
end
# Note that this is nested by default.
function mul_lazy(a1, a2; flatten::Bool = false)
    return flatten ? mul_lazy_flattened(a1, a2) : mul_lazy_nested(a1, a2)
end
mul_lazy_nested(a1, a2) = lazy(Mul([a1, a2]))
to_mul_arguments(a) = ismul(a) ? arguments(a) : [a]
mul_lazy_flattened(a1, a2) = lazy(Mul([to_mul_arguments(a1); to_mul_arguments(a2)]))
mul_lazy(a1::Number, a2) = error("Not implemented.")
mul_lazy(a1, a2::Number) = error("Not implemented.")
mul_lazy(a1::Number, a2::Number) = a1 * a2
div_lazy(a1, a2::Number) = error("Not implemented.")

# NamedDimsArrays.jl interface.
function inds_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return inds(u)
    elseif ismul(u)
        return mapreduce(inds, symdiff, arguments(u))
    else
        return error("Variant not supported.")
    end
end
function dename_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return dename(u)
    else
        return error("Variant not supported.")
    end
end
