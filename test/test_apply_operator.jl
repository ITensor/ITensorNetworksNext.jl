import NamedDimsArrays as NDA
import TensorAlgebra as TA
using GradedArrays: U1, gradedrange
using Graphs: dst, edges, src, vertices
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork, apply_operator, apply_operators,
    beliefpropagation_normnetwork, identity_norm_message_env, ones_norm_message_env
using MatrixAlgebraKit: truncrank
using NamedDimsArrays: name, operator, randname, setname
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_cycle_graph, named_path_graph
using Test: @test, @testset

# Tests run on both a plain (`:nograded`) and a U(1)-graded (`:u1`) backend; the
# array type follows from the axes. On the graded backend the sites are spin-1
# (`U1` charges 2, 0, -2), chosen because the zero-charge sector lets a definite
# total-charge product state be built with every site in that sector and every
# bond trivial, on any graph and with no charge-flux bookkeeping. `random_state`
# seeds that product state and entangles it with random charge-conserving gates.
# Several of the gauging-convention bugs this suite covers only surface on the
# graded backend, so running it over both backends is the regression coverage.

site_axis(::Val{:nograded}, d::Int) = Index(d)
function site_axis(::Val{:u1}, d::Int)
    return Index(gradedrange([U1(c) => 1 for c in (d - 1):-2:(-(d - 1))]))
end

# Trivial (length-1) bond used to seed a product state: charge 0 on the graded
# backend, length 1 on `Base.OneTo`.
trivial_link(::Val{:nograded}) = Index(1)
trivial_link(::Val{:u1}) = Index(gradedrange([U1(0) => 1]))

# Random tensor network on `g`: one named site axis per vertex (`site_axes`) and
# one named link axis per edge (`link_axes`). On graded link axes the two
# endpoints must hold conj-related ranges so the edge contracts; on a
# `Base.OneTo` link `conj` is identity.
function random_tensornetwork(g, link_axes, site_axes)
    function link_axis_at(v, e)
        e_can = haskey(link_axes, e) ? e : reverse(e)
        ax = link_axes[e_can]
        return v == src(e_can) ? ax : conj(ax)
    end
    return TensorNetwork(g) do v
        return randn((site_axes[v], (link_axis_at(v, e) for e in incident_edges(g, v))...))
    end
end

# Random operator acting on `domain_namedaxes`, mapping them to fresh codomain
# names so that `apply` leaves the acted-on dimension names unchanged. The fresh
# names come from `randname` on the dimension *names* (not the axes), which is
# collision-free.
function randn_operator(domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, randname.(name.(domain_namedaxes)))
    # For graded axes the operator's domain side must be conj to its codomain
    # so the gate carries a singlet and contracts cleanly with a non-conj
    # state-site axis. `conj` is identity on `Base.OneTo`.
    dual_domain_namedaxes = setname.(conj.(domain_namedaxes), name.(domain_namedaxes))
    data = randn((codomain_namedaxes..., dual_domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
end

# Random product state: every bond trivial, so on the graded backend each site
# sits in its zero-charge sector and `randn` fills the single allowed block; on
# `:nograded` it is an ordinary random product state.
function product_state(s, g)
    site_axes = Dict(v => site_axis(s, 3) for v in vertices(g))
    link_axes = Dict(e => trivial_link(s) for e in edges(g))
    return random_tensornetwork(g, link_axes, site_axes), site_axes
end

# Entangled definite-charge state: apply layers of random charge-conserving
# 2-site gates to the product state, truncating each split to `maxdim` to throw
# out locally small contributions and keep the bond spectrum well-conditioned.
# The gauge-trivial identity environment makes each gate application exact, and
# reusing `apply_operator` keeps the test self-contained.
function random_state(s, g; nlayers = 2, maxdim = 4)
    state, site_axes = product_state(s, g)
    env = identity_norm_message_env(state)
    for _ in 1:nlayers, e in edges(g)
        gate = randn_operator((site_axes[src(e)], site_axes[dst(e)]))
        state, env = apply_operator(gate, state, env; trunc = truncrank(maxdim))
    end
    return state, site_axes
end

@testset "apply_operator (symmetry = :$sym)" for sym in (:nograded, :u1)
    s = Val(sym)
    N = 4

    # `@testset` reseeds the global RNG on entry to each nested testset, so the
    # state, environment, and gates are built inside each one.
    @testset "untruncated gates are exact (gauge-invariant)" begin
        g = named_cycle_graph(N)
        state, site_axes = random_state(s, g)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
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
        g = named_path_graph(N)
        state, site_axes = random_state(s, g)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        gate = randn_operator((site_axes[2], site_axes[3]))
        # Stays on a path graph (a tree): BP is exact there, so the converged
        # simple-update truncation equals the global optimal SVD oracle, which
        # gates the fully contracted state and truncates rank-`k` across the
        # 2 | 3 cut.
        gated_full = NDA.apply(gate, prod(state))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = TA.svd(gated_full, left; trunc = truncrank(k))
        gated, _ = apply_operator(gate, state, env; trunc = truncrank(k))
        @test prod(gated) ≈ U * S * Vt
    end

    @testset "apply_operators applies a sequence" begin
        g = named_cycle_graph(N)
        state, site_axes = random_state(s, g)
        env = beliefpropagation_normnetwork(
            state, ones_norm_message_env(state);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-13)
        )
        # Gates on neighboring edges sharing site 3, applied in sequence.
        g1 = randn_operator((site_axes[2], site_axes[3]))
        g2 = randn_operator((site_axes[3], site_axes[4]))
        gated, _ = apply_operators([g1, g2], state, env)
        @test prod(gated) ≈ NDA.apply(g2, NDA.apply(g1, prod(state)))
    end
end
