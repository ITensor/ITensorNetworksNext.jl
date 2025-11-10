using AbstractTrees: AbstractTrees
using TermInterface: TermInterface, arguments, iscall, operation
using TypeParameterAccessors: unspecify_type_parameters

# Generic functionality for Applied types, like `Mul`, `Add`, etc.
ismul(a) = iscall(a) && operation(a) ≡ *
head_applied(a) = operation(a)
iscall_applied(a) = true
isexpr_applied(a) = iscall(a)
function show_applied(io::IO, a)
    args = map(arg -> sprint(AbstractTrees.printnode, arg), arguments(a))
    print(io, "(", join(args, " $(operation(a)) "), ")")
    return nothing
end
sorted_arguments_applied(a) = arguments(a)
children_applied(a) = arguments(a)
sorted_children_applied(a) = sorted_arguments(a)
function maketerm_applied(type, head, args, metadata)
    term = type(args)
    @assert head ≡ operation(term)
    return term
end
map_arguments_applied(f, a) = unspecify_type_parameters(typeof(a))(map(f, arguments(a)))
function hash_applied(a, h::UInt64)
    h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
    for arg in arguments(a)
        h = hash(arg, h)
    end
    return h
end

abstract type Applied end
TermInterface.head(a::Applied) = head_applied(a)
TermInterface.iscall(a::Applied) = iscall_applied(a)
TermInterface.isexpr(a::Applied) = isexpr_applied(a)
Base.show(io::IO, a::Applied) = show_applied(io, a)
TermInterface.sorted_arguments(a::Applied) = sorted_arguments_applied(a)
TermInterface.children(a::Applied) = children_applied(a)
TermInterface.sorted_children(a::Applied) = sorted_children_applied(a)
function TermInterface.maketerm(type::Type{<:Applied}, head, args, metadata)
    return maketerm_applied(type, head, args, metadata)
end
map_arguments(f, a::Applied) = map_arguments_applied(f, a)
Base.hash(a::Applied, h::UInt64) = hash_applied(a, h)

struct Mul{A} <: Applied
    arguments::Vector{A}
end
TermInterface.arguments(m::Mul) = getfield(m, :arguments)
TermInterface.operation(m::Mul) = *
