using DataGraphs: underlying_graph
using Graphs: edges
using NamedDimsArrays:
    codomainnames, dimnames, domainnames, name, operator, randname, replacedimnames, state
using NamedGraphs.GraphsExtensions: incident_edges

"""
    normnetwork(tn) -> norm_tn, linknames_map

Build the double-layer norm network `⟨tn|tn⟩` together with the per-edge ket→bra name
mapping used to construct it.

Each ket link axis on every edge is paired with a fresh `randname`-generated bra link
name; the bra layer at every vertex is the ket tensor with all of its incident link
names renamed accordingly. The returned `linknames_map` is keyed by both directions of
each undirected edge (the values are shared `Dict`s, so a directed edge and its reverse
look up the same `ketname => braname` table) and is the source of truth for adapting
externally-supplied messages onto the double-layer network.

Anticipates a future `NormNetwork(tn)` struct that bundles `norm_tn` and `linknames_map`
into a single value with `beliefpropagation` dispatch.
"""
function normnetwork(tn)
    g = underlying_graph(tn)
    linknames_map = Dict()
    for e in edges(tn)
        ket_to_bra = Dict(name(ind) => randname(name(ind)) for ind in linkinds(tn, e))
        linknames_map[e] = ket_to_bra
        linknames_map[reverse(e)] = ket_to_bra
    end
    norm_tn = TensorNetwork(g) do v
        t = tn[v]
        renames = collect(
            Iterators.flatten(linknames_map[e] for e in incident_edges(g, v))
        )
        return t * replacedimnames(t, renames...)
    end
    return norm_tn, linknames_map
end

"""
    beliefpropagation_normnetwork(tn, messages; kwargs...) -> MessageCache

Run belief propagation on the norm network `⟨tn|tn⟩` (treating `tn` as the ket),
starting from a pre-built operator `MessageCache` `messages` (e.g. from
[`identity_norm_messagecache`](@ref) or any of the other `*_norm_messagecache`
constructors).

The norm network built by [`normnetwork`](@ref) is the source of truth for bra-link
names. Each input operator message's domain (bra) axes are renamed to match the
norm-network's bra names before BP iterates; the converged messages are wrapped back as
operators using those same bra names on output. `kwargs` are forwarded to
[`beliefpropagation`](@ref).

Anticipates a future `beliefpropagation(NormNetwork(tn), messages)` once a `NormNetwork`
wrapper type lands; until then this is the canonical entry point for BP on the norm
network.
"""
function beliefpropagation_normnetwork(tn, messages; kwargs...)
    norm_tn, linknames_map = normnetwork(tn)
    raw_messages = Dict(
        e => _retarget_bra(messages[e], linknames_map[e]) for e in keys(messages)
    )
    cache = beliefpropagation(norm_tn, raw_messages; kwargs...)
    return MessageCache(
        Dict(
            e => _wrap_as_norm_operator(cache[e], linknames_map[e])
                for e in keys(cache)
        )
    )
end

# Rename the bra (domain) axes of an operator message to match the supplied
# `ketname => braname` map, returning the underlying named array unwrapped from the
# operator. Codomain names are assumed to be paired one-to-one with domain names in
# the operator's `Bijection` (operator constructor invariant).
function _retarget_bra(op_msg, ket_to_bra)
    raw = state(op_msg)
    renames = Pair[]
    for (kn, current_bn) in zip(codomainnames(op_msg), domainnames(op_msg))
        target_bn = ket_to_bra[kn]
        current_bn == target_bn || push!(renames, current_bn => target_bn)
    end
    return isempty(renames) ? raw : replacedimnames(raw, renames...)
end

# Re-wrap a raw double-layer message as an operator. The codomain names are the ket
# names found in `dimnames(raw)` (a subset of the keys of `ket_to_bra`); the domain
# names are their bra partners.
function _wrap_as_norm_operator(raw, ket_to_bra)
    co_names = Tuple(n for n in dimnames(raw) if haskey(ket_to_bra, n))
    dom_names = map(n -> ket_to_bra[n], co_names)
    return operator(raw, co_names, dom_names)
end
