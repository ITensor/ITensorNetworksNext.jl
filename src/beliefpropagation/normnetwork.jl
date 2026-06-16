using DataGraphs: underlying_graph
using Graphs: dst, edges, edgetype, src
using NamedDimsArrays: codomainnames, denamed, domainnames, name, operator, randname,
    replacedimnames, similar_operator, state
using NamedGraphs.GraphsExtensions: all_edges, incident_edges
using Random: Random, rand!, randn!

# === Norm-network environment constructors ===
#
# `*_norm_message_env(tn)` builds a `MessageCache` shaped to act as the BP environment
# for the norm network ⟨tn|tn⟩, with each entry filled per the leading verb (`identity`,
# `ones`, `randn`, `rand`). The `_env` suffix is reserved for the high-level
# environment-builder interface; the low-level `MessageCache` / `messagecache(...)`
# constructors are used internally. A parallel `*_norm_ctm_env` family is planned for
# CTMRG environments.

"""
    similar_norm_message_env(tn) -> MessageCache

Allocate a BP environment for the norm network ⟨tn|tn⟩ with **undefined** message data:
one square operator message per directed edge of `tn` (both directions on every
undirected edge). On each undirected edge the two directions share the same ket-side
names (the link axes from `tn`) and the same fresh `randname`-generated bra-side names,
with the codomain and domain swapped between the two directions — so `env[v1=>v2]` and
`env[v2=>v1]` contract directly with each other (matching names, dual axes) for
bond-marginal computations. Element type and backend are inherited from the factor
tensors of `tn` via `Base.similar`.

Used internally by [`norm_message_env`](@ref) and the filled environment constructors
([`identity_norm_message_env`](@ref), [`ones_norm_message_env`](@ref),
[`randn_norm_message_env`](@ref), [`rand_norm_message_env`](@ref)). Use it directly to
construct environments with custom message data, e.g. by mutating each entry after
allocation.
"""
function similar_norm_message_env(tn)
    pairs = []
    for e in edges(tn)
        v1, v2 = src(e), dst(e)
        ket_axes = linkinds(tn, e)
        ket_names = name.(ket_axes)
        unnamed_axes = denamed.(ket_axes)
        bra_names = randname.(ket_names)
        # Message axes are dual to the link they contract against in the factor.
        push!(
            pairs,
            edgetype(tn)(v1, v2) =>
                similar_operator(tn[v1], conj.(unnamed_axes), bra_names, ket_names)
        )
        push!(
            pairs,
            edgetype(tn)(v2, v1) =>
                similar_operator(tn[v2], unnamed_axes, bra_names, ket_names)
        )
    end
    return messagecache(pairs)
end

"""
    norm_message_env(f, tn) -> MessageCache

Allocate a norm-network BP environment via [`similar_norm_message_env`](@ref) and apply
`f` to each operator-message entry. Shared building block for the filled-environment
constructors.
"""
function norm_message_env(f, tn)
    env = similar_norm_message_env(tn)
    # TODO: replace with `map(f, env)` once `map` is defined on `MessageCache`.
    foreach(e -> env[e] = f(env[e]), edges(env))
    return env
end

"""
    identity_norm_message_env(tn) -> MessageCache

Build a norm-network BP environment with identity-operator messages on every edge — the
"uncorrelated environment" starting point for belief-propagation simple-update gauging
on ⟨tn|tn⟩.

See also: [`ones_norm_message_env`](@ref), [`randn_norm_message_env`](@ref),
[`rand_norm_message_env`](@ref), [`similar_norm_message_env`](@ref).
"""
identity_norm_message_env(tn) = norm_message_env(one, tn)

"""
    ones_norm_message_env(tn) -> MessageCache

Build a norm-network BP environment whose per-edge messages have every entry equal to
`1` — the rank-1 outer product of all-ones vectors on each (codomain, domain) pair.

See also: [`identity_norm_message_env`](@ref), [`randn_norm_message_env`](@ref),
[`rand_norm_message_env`](@ref).
"""
ones_norm_message_env(tn) = norm_message_env(msg -> fill!(msg, one(eltype(msg))), tn)

randn_norm_message_env(tn) = randn_norm_message_env(Random.default_rng(), tn)

"""
    randn_norm_message_env([rng], tn) -> MessageCache

Build a norm-network BP environment whose per-edge messages have entries drawn from a
standard normal distribution. `rng` defaults to `Random.default_rng()`.

See also: [`rand_norm_message_env`](@ref), [`identity_norm_message_env`](@ref),
[`ones_norm_message_env`](@ref).
"""
function randn_norm_message_env(rng::Random.AbstractRNG, tn)
    return norm_message_env(msg -> randn!(rng, msg), tn)
end

rand_norm_message_env(tn) = rand_norm_message_env(Random.default_rng(), tn)

"""
    rand_norm_message_env([rng], tn) -> MessageCache

Build a norm-network BP environment whose per-edge messages have entries drawn from a
uniform distribution on `[0, 1)`. `rng` defaults to `Random.default_rng()`.

See also: [`randn_norm_message_env`](@ref), [`identity_norm_message_env`](@ref),
[`ones_norm_message_env`](@ref).
"""
function rand_norm_message_env(rng::Random.AbstractRNG, tn)
    return norm_message_env(msg -> rand!(rng, msg), tn)
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
into a single value with belief-propagation dispatch.
"""
function normnetwork(tn)
    linknames_map = Dict(
        e => Dict(n => randname(n) for n in linknames(tn, e))
            for e in edges(tn)
    )
    merge!(linknames_map, Dict(reverse(e) => m for (e, m) in linknames_map))
    norm_tn = TensorNetwork(underlying_graph(tn)) do v
        t = tn[v]
        ket_to_bra = Dict(p for e in incident_edges(tn, v) for p in linknames_map[e])
        return t * replacedimnames(n -> get(ket_to_bra, n, n), conj(t))
    end
    return norm_tn, linknames_map
end

"""
    beliefpropagation_normnetwork(tn, messages; kwargs...) -> MessageCache

Run belief propagation on the norm network `⟨tn|tn⟩` (treating `tn` as the ket),
starting from a pre-built operator `MessageCache` `messages` (e.g. from
[`identity_norm_message_env`](@ref) or any of the other `*_norm_message_env`
constructors).

The norm network built by [`normnetwork`](@ref) is the source of truth for bra-link
names. Each input operator message's codomain (bra) axes are renamed to match the
norm-network's bra names before BP iterates; the converged messages are wrapped back as
operators using those same bra names on output. `kwargs` are forwarded to
`beliefpropagation`.

Anticipates a future `beliefpropagation(NormNetwork(tn), messages)` once a `NormNetwork`
wrapper type lands; until then this is the canonical entry point for BP on the norm
network.
"""
function beliefpropagation_normnetwork(tn, messages; kwargs...)
    norm_tn, linknames_map = normnetwork(tn)

    # Adapt input messages onto the norm network: rename each operator's codomain
    # (bra) axes to the bra names `linknames_map` chose, paired via the operator's
    # own domain (ket) → codomain (bra) bijection.
    es = collect(keys(messages))
    raws = map(es) do e
        msg, ket_to_bra = messages[e], linknames_map[e]
        bra_rename = Dict(
            cur => ket_to_bra[kn] for
                (kn, cur) in zip(domainnames(msg), codomainnames(msg))
        )
        return replacedimnames(n -> get(bra_rename, n, n), state(msg))
    end
    raw_messages = Dict(es .=> raws)

    cache = beliefpropagation(norm_tn, raw_messages; kwargs...)

    # Re-wrap each converged message as an operator with codomain = bra names and
    # domain = ket names from the map.
    return messagecache(keys(cache)) do e
        return operator(cache[e], values(linknames_map[e]), keys(linknames_map[e]))
    end
end
