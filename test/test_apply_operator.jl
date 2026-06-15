import NamedDimsArrays as NDA
import TensorAlgebra as TA
using GradedArrays: U1, gradedrange
using Graphs: edges, src, vertices
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork, apply_operator, apply_operators,
    beliefpropagation_normnetwork, ones_norm_message_env
using MatrixAlgebraKit: truncrank
using NamedDimsArrays: name, operator, randname, setname
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_cycle_graph, named_path_graph
using Test: @test, @testset

# The helpers below are written against the `NamedDimsArrays` interface (named
# axes, `randname`, `operator`, `randn`), so the array type is determined by the
# axes passed in. Each test runs on both `Base.OneTo` (`:nograded`) and
# U(1)-graded (`:u1`) site / link axes, built from `site_axis` / `link_axis`.
# Many of the convention bugs the AKLT validation testbed surfaced
# (`similar_norm_message_env` codomain `isdual`, env-writeback direction swap,
# TA factorization axes) only fail on the graded backend, so running the full
# suite over both backends is the regression coverage in INN proper.

site_axis(::Val{:nograded}, d::Int) = Index(d)
link_axis(::Val{:nograded}, χ::Int) = Index(χ)
function site_axis(::Val{:u1}, d::Int)
    # Even-dim physical: symmetric charges so the on-site spectrum is closed
    # under conj.
    return Index(gradedrange([U1(c) => 1 for c in (d - 1):-2:(-(d - 1))]))
end
function link_axis(::Val{:u1}, χ::Int)
    # `χ` unit sectors carrying charges 0, ±1, ±2, ..., rich enough to contract
    # with the site charges.
    charges = [0]
    c = 1
    while length(charges) < χ
        push!(charges, c, -c)
        c += 1
    end
    return Index(gradedrange([U1(q) => 1 for q in charges[1:χ]]))
end

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

@testset "apply_operator (symmetry = :$sym)" for sym in (:nograded, :u1)
    s = Val(sym)
    N, d, χ = 4, 2, 4

    # `@testset` reseeds the global RNG on entry to every (nested) testset, so we
    # build the network, environment, and gates inside each one. That keeps the
    # link axes as the first draws from each testset's RNG stream, so every later
    # `randname` — the gate codomains here, and the rank names created inside the
    # gate application — stays distinct from the link names.
    @testset "untruncated gates are exact (gauge-invariant)" begin
        g = named_cycle_graph(N)
        link_axes = Dict(e => link_axis(s, χ) for e in edges(g))
        site_axes = Dict(v => site_axis(s, d) for v in vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
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
        link_axes = Dict(e => link_axis(s, χ) for e in edges(g))
        site_axes = Dict(v => site_axis(s, d) for v in vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
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
        link_axes = Dict(e => link_axis(s, χ) for e in edges(g))
        site_axes = Dict(v => site_axis(s, d) for v in vertices(g))
        state = random_tensornetwork(g, link_axes, site_axes)
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
