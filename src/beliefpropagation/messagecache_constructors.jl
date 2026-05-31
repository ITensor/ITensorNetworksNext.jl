using Graphs: edges, src
using NamedDimsArrays: NamedDimsArrays
using Random: Random

# Build a `MessageCache` whose per-edge entry is `f(similar_operator(...))`, with one
# directed edge per direction on every undirected edge of `tn`. The norm-network
# interpretation: each message lives on the (ket, bra) pair for that edge.
#
# `f` decides the message's initial value: `identity` for an uninitialized cache,
# `Base.one` for an identity-filled cache, etc.
function _per_edge_norm_messagecache(f, tn; eltype = _scalartype(tn))
    return messagecache(_all_directed_edges(tn)) do e
        proto = tn[src(e)]
        codomain = Tuple(linkinds(tn, e))
        return f(similar_operator(proto, eltype, codomain))
    end
end

"""
    similar_norm_messagecache(tn; eltype = scalartype(tn)) -> MessageCache

Allocate a `MessageCache` of square operator messages with **undefined** data, one per
directed edge of the undirected graph of `tn` (both directions on every undirected edge).
Each message's codomain is the link axis on that edge in `tn`; the domain has dual
axes with fresh `randname`-generated names.

This is the allocator that backs the filled-cache constructors
(`identity_norm_messagecache`, `ones_norm_messagecache`, `randn_norm_messagecache`).
Use it directly to construct caches with custom message data, e.g. by mutating each
entry after allocation.
"""
function similar_norm_messagecache(tn; kwargs...)
    return _per_edge_norm_messagecache(identity, tn; kwargs...)
end

"""
    identity_norm_messagecache(tn; eltype = scalartype(tn)) -> MessageCache

Allocate a `MessageCache` of identity-operator messages, one per directed edge of `tn`.
Each message acts as the identity map on the link axis for its edge — the
"uncorrelated environment" starting point for belief-propagation simple-update gauging
on the norm network ⟨tn|tn⟩.

See also: [`ones_norm_messagecache`](@ref), [`randn_norm_messagecache`](@ref),
[`similar_norm_messagecache`](@ref).
"""
function identity_norm_messagecache(tn; kwargs...)
    return _per_edge_norm_messagecache(Base.one, tn; kwargs...)
end

"""
    ones_norm_messagecache(tn; eltype = scalartype(tn)) -> MessageCache

Allocate a `MessageCache` whose per-edge messages have every entry equal to `1`. Each
message is the rank-1 outer product of all-ones vectors on the (codomain, domain) link
axes.

See also: [`identity_norm_messagecache`](@ref), [`randn_norm_messagecache`](@ref).
"""
function ones_norm_messagecache(tn; kwargs...)
    return _per_edge_norm_messagecache(
        msg -> Base.fill!(msg, one(eltype(msg))),
        tn;
        kwargs...
    )
end

"""
    randn_norm_messagecache(tn; eltype = scalartype(tn)) -> MessageCache

Allocate a `MessageCache` whose per-edge messages are positive-semidefinite random
matrices `X' * X` with `X` drawn from `randn`. Useful as a non-trivial starting point
for belief-propagation iteration when the converged behavior is expected to be PSD
(e.g. norm-network environments).

See also: [`identity_norm_messagecache`](@ref), [`ones_norm_messagecache`](@ref).
"""
function randn_norm_messagecache(tn; kwargs...)
    return _per_edge_norm_messagecache(tn; kwargs...) do msg
        return _randn_then_gram!(msg)
    end
end

# Fill `msg`'s underlying data with a PSD random matrix `X' * X`, working at the raw
# storage level. Avoids `msg' * msg` at the operator level, which currently breaks on
# ITensor-backed operators whose static `ndims` parameter is `Any` (the `adjoint`
# path requires `ndims` to be statically `Int`). Returns `msg` mutated in place.
function _randn_then_gram!(msg)
    raw = NamedDimsArrays.denamed(NamedDimsArrays.state(msg))
    T = eltype(raw)
    T = T === Any ? Float64 : T
    sz = size(raw)
    K = length(NamedDimsArrays.codomainnames(msg))
    co_dim = prod(ntuple(i -> sz[i], K))
    dom_dim = prod(ntuple(i -> sz[K + i], length(sz) - K))
    X = Random.randn(T, co_dim, dom_dim)
    gram = X' * X
    copyto!(raw, reshape(gram, sz))
    return msg
end

function _scalartype(tn)
    T = eltype(tn[first(vertices(tn))])
    # ITensor-backed tensor networks have `eltype` returning `Any` since storage is
    # dynamic. Fall back to `Float64` so the default constructors produce a usable
    # cache; users with concrete eltypes can pass `eltype = …` explicitly.
    return T === Any ? Float64 : T
end

function _all_directed_edges(tn)
    es = edges(tn)
    return collect(Iterators.flatten(((e, reverse(e)) for e in es)))
end
