using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries
using Graphs: Graphs, AbstractGraph, edges, vertices
using ITensorBase: ITensorBase, NamedTensorOperator, dimnames, dimnametype, inputnames,
    operator, outputnames, state
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

"""
    struct ITensorNetworkOperator{T, V, I, P} <: AbstractITensorNetwork{T, V}

The network equivalent of `ITensorBase.ITensorOperator`: a tensor network of type
`P <: AbstractITensorNetwork{T, V}` together with a pairing of its dangling index names,
where `outputnames[i]` is paired with `inputnames[i]`. Applying the operator contracts over
the input names and leaves the output names.

A pair may straddle two vertices, as it does for a swap or a translation. Indexing returns
the vertex tensor wrapped as an `ITensorOperator` carrying only the pairs whose two halves
both sit on that vertex; the remaining legs are dangling on the wrapper.
"""
struct ITensorNetworkOperator{T, V, I, P <: AbstractITensorNetwork{T, V}} <:
    AbstractITensorNetwork{T, V}
    parent::P
    outputnames::Vector{I}
    inputnames::Vector{I}
    function ITensorNetworkOperator(
            parent::AbstractITensorNetwork{T, V}, outputnames, inputnames
        ) where {T, V}
        I = dimnametype(T)
        outputnames = collect(I, outputnames)
        inputnames = collect(I, inputnames)
        if length(outputnames) != length(inputnames)
            throw(
                ArgumentError(
                    "Operator `outputnames` and `inputnames` must have equal length " *
                        "(positional pairing), got $(length(outputnames)) and " *
                        "$(length(inputnames))."
                )
            )
        end
        for opname in Iterators.flatten((outputnames, inputnames))
            nvertices = length(dimnamevertices(parent, opname))
            if nvertices != 1
                throw(
                    ArgumentError(
                        "operator dim name $opname is associated with $nvertices vertices " *
                            "in the tensor network; an operator leg must be a dangling index."
                    )
                )
            end
        end
        return new{T, V, I, typeof(parent)}(parent, outputnames, inputnames)
    end
end

function ITensorBase.operator(tn::AbstractITensorNetwork, outputnames, inputnames)
    return ITensorNetworkOperator(tn, outputnames, inputnames)
end

function Base.eltype(::Type{<:ITensorNetworkOperator{T, V, I}}) where {T, V, I}
    return NamedTensorOperator{I, T}
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(op::ITensorNetworkOperator) = edges(state(op))
Graphs.vertices(op::ITensorNetworkOperator) = vertices(state(op))

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(op::ITensorNetworkOperator, vertex)
    return encoded_vertex(state(op), vertex)
end
function NamedGraphs.decoded_vertex(op::ITensorNetworkOperator, code::Integer)
    return decoded_vertex(state(op), code)
end
NamedGraphs.encoded_graph(op::ITensorNetworkOperator) = encoded_graph(state(op))

# ==================================== DataGraphs.jl ===================================== #

# A pair whose two halves sit on different vertices has no bijection to give this vertex, so
# only the pairs local to `vertex` become the wrapper's pairing and the rest stay dangling.
function DataGraphs.get_vertex_data(op::ITensorNetworkOperator, vertex)
    tensor = state(op)[vertex]
    names = dimnames(tensor)
    outputs = similar(outputnames(op), 0)
    inputs = similar(inputnames(op), 0)
    for (output, input) in zip(outputnames(op), inputnames(op))
        if output in names && input in names
            push!(outputs, output)
            push!(inputs, input)
        end
    end
    return operator(tensor, outputs, inputs)
end

function DataGraphs.is_vertex_assigned(op::ITensorNetworkOperator, vertex)
    return is_vertex_assigned(state(op), vertex)
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::ITensorNetworkOperator) = false
Dictionaries.isinsertable(::ITensorNetworkOperator) = false

# ====================================== interface ======================================= #

ITensorBase.state(op::ITensorNetworkOperator) = op.parent
Base.parent(op::ITensorNetworkOperator) = state(op)
ITensorBase.outputnames(op::ITensorNetworkOperator) = op.outputnames
ITensorBase.inputnames(op::ITensorNetworkOperator) = op.inputnames

function supportof(tn::AbstractGraph, op::ITensorNetworkOperator)
    return supportof_dimnames(tn, inputnames(op))
end
