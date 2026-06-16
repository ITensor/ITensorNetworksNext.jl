import NamedDimsArrays as NDA
import TensorAlgebra as TA
using GradedArrays: U1, gradedrange
using Graphs: dst, edges, src, vertices
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork, apply_operator, apply_operators,
    beliefpropagation_normnetwork, identity_norm_message_env, insertlink!,
    ones_norm_message_env
using MatrixAlgebraKit: truncrank
using NamedDimsArrays: name, operator, randname, setname
using NamedGraphs.NamedGraphGenerators: named_cycle_graph, named_path_graph
using NamedGraphs: NamedGraph
using Test: @test, @testset

const spinone = Base.OneTo(3)
const spinone_u1 = gradedrange([U1(2) => 1, U1(0) => 1, U1(-2) => 1])

function randn_operator(domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, randname.(name.(domain_namedaxes)))
    dual_domain_namedaxes = setname.(conj.(domain_namedaxes), name.(domain_namedaxes))
    data = randn((codomain_namedaxes..., dual_domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
end

function random_state(g, site_axes; nlayers = 2, trunc = truncrank(4))
    state = TensorNetwork(NamedGraph(collect(vertices(g)))) do v
        return randn((site_axes[v],))
    end
    for e in edges(g)
        insertlink!(state, e)
    end
    env = identity_norm_message_env(state)
    for _ in 1:nlayers, e in edges(g)
        gate = randn_operator((site_axes[src(e)], site_axes[dst(e)]))
        state, env = apply_operator(gate, state, env; trunc)
    end
    return state
end

@testset "apply_operator ($(nameof(typeof(site_range))))" for site_range in (
        spinone, spinone_u1,
    )
    N = 4

    @testset "untruncated gates are exact (gauge-invariant)" begin
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        state = random_state(g, site_axes)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        for gate in (
                randn_operator((site_axes[2],)),
                randn_operator((site_axes[2], site_axes[3])),
            )
            gated, _ = apply_operator(gate, state, env)
            @test prod(gated) ≈ NDA.apply(gate, prod(state))
        end
    end

    @testset "truncated 2-site gate matches global optimal SVD (rank $k)" for k in 1:3
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        state = random_state(g, site_axes)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        gate = randn_operator((site_axes[2], site_axes[3]))
        gated_full = NDA.apply(gate, prod(state))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = TA.svd(gated_full, left; trunc = truncrank(k))
        gated, _ = apply_operator(gate, state, env; trunc = truncrank(k))
        @test prod(gated) ≈ U * S * Vt
    end

    @testset "apply_operators applies a sequence" begin
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        state = random_state(g, site_axes)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        g1 = randn_operator((site_axes[2], site_axes[3]))
        g2 = randn_operator((site_axes[3], site_axes[4]))
        gated, _ = apply_operators([g1, g2], state, env)
        @test prod(gated) ≈ NDA.apply(g2, NDA.apply(g1, prod(state)))
    end
end
