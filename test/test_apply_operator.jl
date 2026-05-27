import Graphs
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using DataGraphs: underlying_graph
using ITensorBase: Index
using ITensorNetworksNext: MessageCache, TensorNetwork, apply_operator, apply_operators,
    beliefpropagation, linkinds
using MatrixAlgebraKit: truncrank
using NamedDimsArrays: name, operator, randname, replacedimnames, setname
using NamedGraphs.GraphsExtensions: all_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_path_graph
using Test: @test, @testset

# The helpers below are written against the `NamedDimsArrays` interface (named
# axes, `randname`, `operator`, `randn`), so the array type is determined by the
# axes passed in. Here we use ITensor `Index`es.

# Random tensor network on `g`: one named site axis per vertex (`site_axes`) and
# one named link axis per edge (`link_axes`).
function random_tensornetwork(g, link_axes, site_axes)
    link_axis(e) = haskey(link_axes, e) ? link_axes[e] : link_axes[reverse(e)]
    return TensorNetwork(g) do v
        return randn((site_axes[v], (link_axis(e) for e in incident_edges(g, v))...))
    end
end

# Random operator acting on `domain_namedaxes`, mapping them to fresh codomain
# names so that `apply` leaves the acted-on dimension names unchanged. The fresh
# names come from `randname` on the dimension *names* (not the axes), which is
# collision-free.
function rand_operator(domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, randname.(name.(domain_namedaxes)))
    data = randn((codomain_namedaxes..., domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
end

# Converged belief-propagation messages on the double-layer norm network
# ⟨state|state⟩: the bra layer's link axes get fresh names so they stay distinct
# from the ket's, while the shared site axis is contracted. Returned as operator
# messages whose codomain is the ket link and whose domain is the bra link. On a
# tree these are the exact bond environments, so the resulting gauge reproduces
# exact (canonical-form) truncation. Anticipates a future
# `beliefpropagation(NormNetwork(state))`. Forwards `kwargs` to `beliefpropagation`.
function beliefpropagation_normnetwork(state; kwargs...)
    g = underlying_graph(state)
    link_name(e) = name(only(linkinds(state, e)))
    bra_name = Dict(link_name(e) => randname(link_name(e)) for e in all_edges(g))
    norm_tn = TensorNetwork(g) do v
        t = state[v]
        bra = [link_name(e) => bra_name[link_name(e)] for e in incident_edges(g, v)]
        return t * replacedimnames(t, bra...)
    end
    init = Dict(e => ones(Float64, Tuple(linkinds(norm_tn, e))) for e in all_edges(g))
    cache = beliefpropagation(norm_tn, init; kwargs...)
    return MessageCache(
        Dict(
            e => operator(cache[e], (link_name(e),), (bra_name[link_name(e)],))
                for e in all_edges(g)
        )
    )
end

@testset "apply_operator on a path graph" begin
    N, χ, d = 4, 4, 2
    g = named_path_graph(N)

    # `@testset` reseeds the global RNG on entry to every (nested) testset, so we
    # build the network, environment, and gates inside each one. That keeps the
    # link `Index`es as the first draws from each testset's RNG stream, so every
    # later `randname` — the gate codomains here, and the rank names created
    # inside the gate application — stays distinct from the link names.
    @testset "untruncated gates are exact (gauge-invariant)" begin
        link_axes = Dict(e => Index(χ) for e in Graphs.edges(g))
        site_axes = Dict(v => Index(d) for v in Graphs.vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
        env = beliefpropagation_normnetwork(
            state; stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        # Without truncation the gate is applied exactly, so the gated network
        # reproduces exact contraction regardless of the gauge.
        for gate in (
                rand_operator((site_axes[2],)),
                rand_operator((site_axes[2], site_axes[3])),
            )
            gated, _ = apply_operator(gate, state, env)
            @test prod(gated) ≈ NDA.apply(gate, prod(state))
        end
    end

    @testset "truncated 2-site gate matches global optimal SVD (rank $k)" for k in 1:3
        link_axes = Dict(e => Index(χ) for e in Graphs.edges(g))
        site_axes = Dict(v => Index(d) for v in Graphs.vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
        env = beliefpropagation_normnetwork(
            state; stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        gate = rand_operator((site_axes[2], site_axes[3]))
        # Exact oracle: gate the fully contracted state, then take the globally
        # optimal rank-`k` SVD truncation across the 2 | 3 cut.
        Ψ = NDA.apply(gate, prod(state))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = TA.svd(Ψ, left; trunc = truncrank(k))
        gated, _ = apply_operator(gate, state, env; trunc = truncrank(k))
        @test prod(gated) ≈ U * S * Vt
    end

    @testset "apply_operators applies a sequence" begin
        link_axes = Dict(e => Index(χ) for e in Graphs.edges(g))
        site_axes = Dict(v => Index(d) for v in Graphs.vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
        env = beliefpropagation_normnetwork(
            state; stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        # Gates on neighboring edges sharing site 3, applied in sequence.
        gA = rand_operator((site_axes[2], site_axes[3]))
        gB = rand_operator((site_axes[3], site_axes[4]))
        gated, _ = apply_operators([gA, gB], state, env)
        @test prod(gated) ≈ NDA.apply(gB, NDA.apply(gA, prod(state)))
    end
end
