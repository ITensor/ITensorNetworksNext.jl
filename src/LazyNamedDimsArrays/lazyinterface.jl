using NamedDimsArrays: denamed, dimnames, inds
using TermInterface: iscall, maketerm, operation, sorted_arguments
using WrappedUnions: unwrap

lazy(x) = error("Not defined.")

# Walk the expression `ex`, modifying the
# operations by `opmap` and the arguments by `argmap`.
function walk(opmap, argmap, ex)
    if !iscall(ex)
        return argmap(ex)
    else
        return mapfoldl(opmap(operation(ex)), arguments(ex)) do (args...)
            return walk(opmap, argmap, args...)
        end
    end
end
# Walk the expression `ex`, modifying the
# operations by `opmap`.
opwalk(opmap, a) = walk(opmap, identity, a)
# Walk the expression `ex`, modifying the
# arguments by `argmap`.
argwalk(argmap, a) = walk(identity, argmap, a)

# Generic lazy functionality.
using FunctionImplementations: AbstractArrayImplementationStyle
struct LazyNamedDimsArrayImplementationStyle <: AbstractArrayImplementationStyle end
const lazy_style = LazyNamedDimsArrayImplementationStyle()

const maketerm_lazy = lazy_style(maketerm)
function maketerm_lazy(type::Type, head, args, metadata)
    if head ≡ *
        return type(maketerm(Mul, head, args, metadata))
    else
        return error("Only mul supported right now.")
    end
end
const getindex_lazy = lazy_style(getindex)
function getindex_lazy(a::AbstractArray, I...)
    u = unwrap(a)
    if !iscall(u)
        return u[I...]
    else
        return error("Indexing into expression not supported.")
    end
end
const arguments_lazy = lazy_style(arguments)
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
using TermInterface: children
const children_lazy = lazy_style(children)
children_lazy(a) = arguments(a)
using TermInterface: head
const head_lazy = lazy_style(head)
head_lazy(a) = operation(a)
const iscall_lazy = lazy_style(iscall)
iscall_lazy(a) = iscall(unwrap(a))
using TermInterface: isexpr
const isexpr_lazy = lazy_style(isexpr)
isexpr_lazy(a) = iscall(a)
const operation_lazy = lazy_style(operation)
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
const sorted_arguments_lazy = lazy_style(sorted_arguments)
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
using TermInterface: sorted_children
const sorted_children_lazy = lazy_style(sorted_children)
sorted_children_lazy(a) = sorted_arguments(a)
const ismul_lazy = lazy_style(ismul)
ismul_lazy(a) = ismul(unwrap(a))
using AbstractTrees: AbstractTrees
const abstracttrees_children_lazy = lazy_style(AbstractTrees.children)
function abstracttrees_children_lazy(a)
    if !iscall(a)
        return ()
    else
        return arguments(a)
    end
end
using AbstractTrees: nodevalue
const nodevalue_lazy = lazy_style(nodevalue)
function nodevalue_lazy(a)
    if !iscall(a)
        return unwrap(a)
    else
        return operation(a)
    end
end
using Base.Broadcast: materialize
const materialize_lazy = lazy_style(materialize)
materialize_lazy(a) = argwalk(unwrap, a)
const copy_lazy = lazy_style(copy)
copy_lazy(a) = materialize(a)
const equals_lazy = lazy_style(==)
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
const isequal_lazy = lazy_style(isequal)
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
const hash_lazy = lazy_style(hash)
function hash_lazy(a, h::UInt64)
    h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
    # Use `_hash`, which defines a custom hash for NamedDimsArray.
    return _hash(unwrap(a), h)
end
const map_arguments_lazy = lazy_style(map_arguments)
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
function substitute end
const substitute_lazy = lazy_style(substitute)
function substitute_lazy(a, substitutions::AbstractDict)
    haskey(substitutions, a) && return substitutions[a]
    !iscall(a) && return a
    return map_arguments(arg -> substitute(arg, substitutions), a)
end
substitute_lazy(a, substitutions) = substitute(a, Dict(substitutions))
using AbstractTrees: printnode
const printnode_lazy = lazy_style(printnode)
function printnode_lazy(io, a)
    # Use `printnode_nameddims` to avoid type piracy,
    # since it overloads on `AbstractNamedDimsArray`.
    return printnode_nameddims(io, unwrap(a))
end
const show_lazy = lazy_style(show)
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
const add_lazy = lazy_style(+)
add_lazy(a1, a2) = error("Not implemented.")
const sub_lazy = lazy_style(-)
sub_lazy(a) = error("Not implemented.")
sub_lazy(a1, a2) = error("Not implemented.")
const mul_lazy = lazy_style(*)
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
const dimnames_lazy = lazy_style(dimnames)
function dimnames_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return dimnames(u)
    elseif ismul(u)
        return mapreduce(dimnames, symdiff, arguments(u))
    else
        return error("Variant not supported.")
    end
end
const inds_lazy = lazy_style(inds)
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
const denamed_lazy = lazy_style(denamed)
function denamed_lazy(a)
    u = unwrap(a)
    if !iscall(u)
        return denamed(u)
    else
        return error("Variant not supported.")
    end
end
