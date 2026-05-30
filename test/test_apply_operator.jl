import Graphs
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork, apply_operator, apply_operators,
    beliefpropagation_normnetwork, identity_norm_messagecache, ones_norm_messagecache,
    randn_norm_messagecache, similar_norm_messagecache
using MatrixAlgebraKit: truncrank
using NamedDimsArrays: name, operator, randname, setname
using NamedGraphs.GraphsExtensions: incident_edges
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
function randn_operator(domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, randname.(name.(domain_namedaxes)))
    data = randn((codomain_namedaxes..., domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
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
                randn_operator((site_axes[2],)),
                randn_operator((site_axes[2], site_axes[3])),
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
        gate = randn_operator((site_axes[2], site_axes[3]))
        # Exact oracle: gate the fully contracted state, then take the globally
        # optimal rank-`k` SVD truncation across the 2 | 3 cut.
        gated_full = NDA.apply(gate, prod(state))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = TA.svd(gated_full, left; trunc = truncrank(k))
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
        g1 = randn_operator((site_axes[2], site_axes[3]))
        g2 = randn_operator((site_axes[3], site_axes[4]))
        gated, _ = apply_operators([g1, g2], state, env)
        @test prod(gated) ≈ NDA.apply(g2, NDA.apply(g1, prod(state)))
    end

    @testset "norm-messagecache constructors" begin
        link_axes = Dict(e => Index(χ) for e in Graphs.edges(g))
        site_axes = Dict(v => Index(d) for v in Graphs.vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)

        # All three constructors build a `MessageCache` with two directed edges per
        # undirected edge of the state.
        n_directed = 2 * length(collect(Graphs.edges(g)))
        for ctor in (
                similar_norm_messagecache, identity_norm_messagecache,
                ones_norm_messagecache, randn_norm_messagecache,
            )
            cache = ctor(state)
            @test length(collect(Graphs.edges(cache))) == n_directed
        end

        # Identity env reproduces the gauge-invariant exact-gate property: an
        # untruncated gate gives the exact result regardless of which valid env we
        # gauge against.
        env = identity_norm_messagecache(state)
        for gate in (
                randn_operator((site_axes[2],)),
                randn_operator((site_axes[2], site_axes[3])),
            )
            gated, _ = apply_operator(gate, state, env)
            @test prod(gated) ≈ NDA.apply(gate, prod(state))
        end
    end
end
