using DataGraphs: underlying_graph
using NamedDimsArrays: NamedDimsArrays
using NamedGraphs.GraphsExtensions: all_edges, incident_edges

"""
    beliefpropagation_normnetwork(tn; eltype = scalartype(tn), kwargs...) -> MessageCache

Run belief propagation on the norm network `⟨tn|tn⟩`, treating `tn` as the ket.

Eagerly builds the double-layer network by contracting each ket tensor with its
bra partner (site axes contracted; bra link axes given fresh `randname`s so they
stay distinct from the ket links), runs [`beliefpropagation`](@ref) on the
resulting scalar network with all-ones initial messages, and converts the
converged per-edge messages to square operators whose codomain is the ket link
and domain is the bra link. The returned cache is directly usable as the BP
environment for `apply_operator` / `apply_operators`.

Anticipates a future `beliefpropagation(NormNetwork(tn))` once a `NormNetwork`
wrapper type lands; until then this is the canonical way to converge BP messages
for the norm network. `kwargs` are forwarded to `beliefpropagation` (e.g.
`stopping_criterion`).
"""
function beliefpropagation_normnetwork(tn; eltype = _scalartype(tn), kwargs...)
    g = underlying_graph(tn)
    link_name(e) = NamedDimsArrays.name(only(linkinds(tn, e)))
    bra_name =
        Dict(link_name(e) => NamedDimsArrays.randname(link_name(e)) for e in all_edges(g))
    norm_tn = TensorNetwork(g) do v
        t = tn[v]
        bra = [link_name(e) => bra_name[link_name(e)] for e in incident_edges(g, v)]
        return t * NamedDimsArrays.replacedimnames(t, bra...)
    end
    init = Dict(e => ones(eltype, Tuple(linkinds(norm_tn, e))) for e in all_edges(g))
    cache = beliefpropagation(norm_tn, init; kwargs...)
    return MessageCache(
        Dict(
            e => NamedDimsArrays.operator(
                    cache[e], (link_name(e),), (bra_name[link_name(e)],)
                )
                for e in all_edges(g)
        )
    )
end
