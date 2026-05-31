using DataGraphs: underlying_graph
using Graphs: edges, src
using NamedDimsArrays:
    codomainnames, denamed, domainnames, name, operator, randname, replacedimnames, state
using NamedGraphs.GraphsExtensions: all_edges, incident_edges
using Random: Random

# === MessageCache constructors keyed to the norm network ⟨tn|tn⟩ ===

"""
    similar_norm_messagecache(tn) -> MessageCache

Allocate a `MessageCache` of square operator messages with **undefined** data, one per
directed edge of the undirected graph of `tn` (both directions on every undirected edge).
Each message's codomain is the link axes on that edge in `tn`; the domain has dual axes
with fresh `randname`-generated names. The element type and backend are inherited from
the factor tensors of `tn` via `Base.similar`.

This is the allocator that backs the filled-cache constructors
(`identity_norm_messagecache`, `ones_norm_messagecache`, `randn_norm_messagecache`).
Use it directly to construct caches with custom message data, e.g. by mutating each
entry after allocation.
"""
function similar_norm_messagecache(tn)
    return messagecache(all_edges(tn)) do e
        return similar_operator(tn[src(e)], linkinds(tn, e))
    end
end

"""
    identity_norm_messagecache(tn) -> MessageCache

Allocate a `MessageCache` of identity-operator messages, one per directed edge of `tn`.
Each message acts as the identity map on the link axis for its edge — the
"uncorrelated environment" starting point for belief-propagation simple-update gauging
on the norm network ⟨tn|tn⟩.

See also: [`ones_norm_messagecache`](@ref), [`randn_norm_messagecache`](@ref),
[`similar_norm_messagecache`](@ref).
"""
function identity_norm_messagecache(tn)
    m = similar_norm_messagecache(tn)
    # TODO: replace with `map(one, m)` once `map` is defined on `MessageCache`.
    foreach(e -> m[e] = one(m[e]), edges(m))
    return m
end

"""
    ones_norm_messagecache(tn) -> MessageCache

Allocate a `MessageCache` whose per-edge messages have every entry equal to `1`. Each
message is the rank-1 outer product of all-ones vectors on the (codomain, domain) link
axes.

See also: [`identity_norm_messagecache`](@ref), [`randn_norm_messagecache`](@ref).
"""
function ones_norm_messagecache(tn)
    m = similar_norm_messagecache(tn)
    # TODO: replace with `map(msg -> fill!(msg, one(eltype(msg))), m)` once `map`
    # is defined on `MessageCache`.
    foreach(e -> m[e] = fill!(m[e], one(eltype(m[e]))), edges(m))
    return m
end

"""
    randn_norm_messagecache(tn) -> MessageCache

Allocate a `MessageCache` whose per-edge messages have entries drawn from `randn`.

See also: [`identity_norm_messagecache`](@ref), [`ones_norm_messagecache`](@ref).
"""
function randn_norm_messagecache(tn)
    m = similar_norm_messagecache(tn)
    # TODO: replace with `map(Random.randn!, m)` once `map` is defined on `MessageCache`.
    # `Random.randn!(m[e])` directly does not work on ITensor-backed operators because
    # `eltype(typeof(::ITensor)) === Any`, which makes `Random.randn!` dispatch on
    # `Type{Any}`; peel to the concrete storage so it sees the runtime eltype.
    foreach(e -> Random.randn!(denamed(state(m[e]))), edges(m))
    return m
end

# === Double-layer construction and BP wrapper ===

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
    linknames_map = Dict(
        e => Dict(name(ind) => randname(name(ind)) for ind in linkinds(tn, e))
            for e in edges(tn)
    )
    merge!(linknames_map, Dict(reverse(e) => m for (e, m) in linknames_map))
    norm_tn = TensorNetwork(underlying_graph(tn)) do v
        t = tn[v]
        ket_to_bra = Dict(p for e in incident_edges(tn, v) for p in linknames_map[e])
        return t * replacedimnames(n -> get(ket_to_bra, n, n), dag(t))
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

    # Adapt input messages onto the norm network: rename each operator's domain (bra)
    # axes to the bra names `linknames_map` chose, paired via the operator's own
    # codomain → domain bijection.
    es = collect(keys(messages))
    raws = map(es) do e
        msg, ket_to_bra = messages[e], linknames_map[e]
        bra_rename = Dict(
            cur => ket_to_bra[kn] for
                (kn, cur) in zip(codomainnames(msg), domainnames(msg))
        )
        return replacedimnames(n -> get(bra_rename, n, n), state(msg))
    end
    raw_messages = Dict(es .=> raws)

    cache = beliefpropagation(norm_tn, raw_messages; kwargs...)

    # Re-wrap each converged message as an operator with codomain = ket names and
    # domain = paired bra names from the map.
    return messagecache(keys(cache)) do e
        return operator(cache[e], keys(linknames_map[e]), values(linknames_map[e]))
    end
end
