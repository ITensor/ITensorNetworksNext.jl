using DataGraphs: DataGraphs, underlying_graph
using Graphs: dst, edgetype, neighbors, src, vertices
using ITensorBase: dimnames, lazy, operator, replacedimnames, state
using NamedGraphs.GraphsExtensions: vertextype

# A lazy `⟨ψ|H|ψ⟩` network: a ket `ITensorNetwork`, a `TensorNetworkOperator` (which carries
# the ket → bra *site* name map), and a forward ket → bra map for the *link* names. The bra
# layer is derived from the ket (`conj` + index renaming), never stored, so updating a ket
# tensor is reflected in the bra. As an `AbstractITensorNetwork`, the data on vertex `v` is
# the lazy product `lazy(ket) * lazy(operator) * lazy(bra)`, so the existing
# `contract_network` / `MessageCache` machinery treats it like any other tensor network.
struct QuadraticFormNetwork{V, VD, Ket, Operator, LinkMap} <:
    AbstractITensorNetwork{V, VD}
    ket::Ket
    operator::Operator
    link_index_map::LinkMap
end

function bra_name_map(qf::QuadraticFormNetwork)
    return merge(site_index_map(qf.operator), qf.link_index_map)
end

ket_tensor(qf::QuadraticFormNetwork, v) = qf.ket[v]
operator_tensor(qf::QuadraticFormNetwork, v) = qf.operator[v]
function bra_tensor(qf::QuadraticFormNetwork, v)
    m = bra_name_map(qf)
    return replacedimnames(n -> get(m, n, n), conj(qf.ket[v]))
end

# === AbstractTensorNetwork / DataGraphs interface ===

DataGraphs.underlying_graph(qf::QuadraticFormNetwork) = underlying_graph(qf.ket)

function DataGraphs.is_vertex_assigned(qf::QuadraticFormNetwork, v)
    return DataGraphs.is_vertex_assigned(qf.ket, v)
end
DataGraphs.is_edge_assigned(::QuadraticFormNetwork, _e) = false

function DataGraphs.get_vertex_data(qf::QuadraticFormNetwork, v)
    return lazy(ket_tensor(qf, v)) * lazy(operator_tensor(qf, v)) * lazy(bra_tensor(qf, v))
end

function Base.copy(qf::QuadraticFormNetwork)
    return QuadraticFormNetwork(copy(qf.ket), copy(qf.operator), qf.link_index_map)
end

# === constructor ===

function QuadraticFormNetwork(ket, operator::TensorNetworkOperator, link_index_map)
    V = vertextype(ket)
    tmp = QuadraticFormNetwork{
        V, Any, typeof(ket), typeof(operator), typeof(link_index_map),
    }(
        ket,
        operator,
        link_index_map
    )
    VD = typeof(DataGraphs.get_vertex_data(tmp, first(vertices(ket))))
    return QuadraticFormNetwork{
        V, VD, typeof(ket), typeof(operator), typeof(link_index_map),
    }(
        ket,
        operator,
        link_index_map
    )
end

# === Environments ===
#
# On a tree, the projected-Hamiltonian environment of `⟨ψ|H|ψ⟩` is exact: the message on
# directed edge `v → w` is the contraction of the entire `⟨ψ|H|ψ⟩` subtree on `v`'s side
# of the cut `(v, w)`. It carries the three bonds crossing the cut (ket, operator, bra).
#
# The messages are computed by a single dependency-ordered pass over every directed edge
# (`forest_cover_edge_sequence` visits each tree edge in both directions, sources before
# targets), with each message an exact contraction of `qf[v]` against the already-computed
# incoming messages from `v`'s other neighbors. No fixed-point iteration is needed.

function incoming_subtree_messages(messages, graph, v, w)
    return [
        state(messages[edgetype(graph)(u, v)]) for
            u in neighbors(graph, v) if u != w
    ]
end

function environment_operator(message, link_index_map)
    ketnames = [n for n in dimnames(message) if haskey(link_index_map, n)]
    branames = [link_index_map[n] for n in ketnames]
    return operator(message, branames, ketnames)
end

"""
    quadratic_form_environments(qf::QuadraticFormNetwork; root) -> MessageCache

Exact projected-Hamiltonian environments of `⟨ψ|H|ψ⟩` on a tree, as a `MessageCache` of
ITensor operators keyed by directed edges. The message on `v → w` is the
contraction of the `⟨ψ|H|ψ⟩` subtree on `v`'s side of `(v, w)`, wrapped as an operator
recording the bra ↔ ket link correspondence (see [`environment_operator`](@ref)).
"""
function quadratic_form_environments(qf::QuadraticFormNetwork; kwargs...)
    sequence = forest_cover_edge_sequence(qf; kwargs...)
    messages = Dict{edgetype(qf), Any}()
    for e in sequence
        v, w = src(e), dst(e)
        incoming = incoming_subtree_messages(messages, qf, v, w)
        message = contract_network([[qf[v]]; incoming])
        messages[e] = environment_operator(message, qf.link_index_map)
    end
    return messagecache(messages)
end
