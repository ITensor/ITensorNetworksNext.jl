# MAK-style algorithm selection helpers (cf. `MatrixAlgebraKit.select_algorithm`
# / `default_algorithm`), but with selection-relevant inputs packed into an
# `args` tuple so the value and type domains stay disjoint: `(1.2,)` vs
# `Tuple{Float64}`. Strategy types subtype `AbstractAlgorithm` so the passthrough
# overload is generic.

abstract type AbstractAlgorithm end

function default_algorithm(f, ::Type{Args}; kwargs...) where {Args <: Tuple}
    return throw(MethodError(default_algorithm, (f, Args)))
end
function default_algorithm(f, args::Tuple; kwargs...)
    return default_algorithm(f, typeof(args); kwargs...)
end

function select_algorithm(f, alg, args::Tuple; kwargs...)
    return select_algorithm(f, alg, typeof(args); kwargs...)
end
function select_algorithm(f, ::Nothing, args::Tuple; kwargs...)
    return default_algorithm(f, args; kwargs...)
end
function select_algorithm(f, alg::NamedTuple, args::Tuple; kwargs...)
    isempty(kwargs) || throw(
        ArgumentError(
            "Additional keyword arguments are not allowed when `alg` is a `NamedTuple`."
        )
    )
    return default_algorithm(f, args; alg...)
end
function select_algorithm(f, ::Nothing, ::Type{Args}; kwargs...) where {Args <: Tuple}
    return default_algorithm(f, Args; kwargs...)
end
function select_algorithm(f, alg::NamedTuple, ::Type{Args}; kwargs...) where {Args <: Tuple}
    isempty(kwargs) || throw(
        ArgumentError(
            "Additional keyword arguments are not allowed when `alg` is a `NamedTuple`."
        )
    )
    return default_algorithm(f, Args; alg...)
end
function select_algorithm(f, alg::AbstractAlgorithm, ::Type{<:Tuple}; kwargs...)
    isempty(kwargs) || throw(
        ArgumentError(
            "Additional keyword arguments are not allowed when `alg` is an `AbstractAlgorithm` instance."
        )
    )
    return alg
end
function select_algorithm(f, alg::AbstractAlgorithm, args::Tuple; kwargs...)
    return select_algorithm(f, alg, typeof(args); kwargs...)
end

# Allocate the destination for an in-place call to `f`. Operations overload
# `initialize_output(::typeof(f), args...)` to control allocation.
function initialize_output(f, args...; kwargs...)
    return throw(MethodError(initialize_output, (f, args...)))
end
