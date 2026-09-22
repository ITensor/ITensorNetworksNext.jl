using GradedArrays: U1, gradedrange
using Graphs: dst, edges, src, vertices
using ITensorBase: Index, apply, name, operator, setname, uniquename
using ITensorNetworksNext: NormNetwork, apply_operator, apply_operators, insertlink!,
    message_environment, tensornetwork
using MatrixAlgebraKit: svd_trunc, truncrank
using NamedGraphs: named_cycle_graph, named_path_graph
using Random: AbstractRNG
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra: sqrth_invsqrth_safe
using TensorKitSectors: FermionParity
using Test: @test, @testset

const spinone = Base.OneTo(3)
const spinone_u1 = gradedrange([U1(2) => 1, U1(0) => 1, U1(-2) => 1])
const fermion = gradedrange([FermionParity(0) => 2, FermionParity(1) => 2])

function randn_operator(rng::AbstractRNG, elt::Type, domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, uniquename.(name.(domain_namedaxes)))
    dual_domain_namedaxes = setname.(conj.(domain_namedaxes), name.(domain_namedaxes))
    data = randn(rng, elt, (codomain_namedaxes..., dual_domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
end

# Build a random state by applying random gates layer by layer, carrying the belief
# propagation environment through the applications. The returned `env` is the environment
# the gate applications produced, ready to gauge the next application (belief-propagation
# convergence itself is covered separately in `test_beliefpropagation.jl`).
function random_state(rng::AbstractRNG, elt::Type, g, site_axes; nlayers, trunc)
    network = tensornetwork(vertices(g)) do v
        return randn(rng, elt, (site_axes[v],))
    end

    for edge in edges(g)
        insertlink!(network, edge)
    end

    env = message_environment(one, NormNetwork(network))
    for _ in 1:nlayers, e in edges(g)
        gate = randn_operator(rng, elt, (site_axes[src(e)], site_axes[dst(e)]))
        network, env = apply_operator(gate, network, env; trunc)
    end
    return network, env
end

@testset "apply_operator (T=$T, $label)" for (label, site_range) in (
            "spinone" => spinone, "spinone_u1" => spinone_u1, "fermion" => fermion,
        ),
        T in (Float32, Float64, ComplexF64)

    N = 4

    @testset "untruncated gates are exact (gauge-invariant)" begin
        rng = StableRNG(123)
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        for gate in (
                randn_operator(rng, T, (site_axes[2],)),
                randn_operator(rng, T, (site_axes[2], site_axes[3])),
            )
            gated, _ = apply_operator(gate, network, env)
            @test prod(gated) ≈ apply(gate, prod(network)) rtol = eps(real(T))^(1 / 3)
        end
    end

    @testset "truncated 2-site gate matches global optimal SVD (rank $k)" for k in 1:3
        rng = StableRNG(123)
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        gate = randn_operator(rng, T, (site_axes[2], site_axes[3]))
        gated_full = apply(gate, prod(network))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = svd_trunc(gated_full, left; trunc = truncrank(k))
        gated, _ = apply_operator(gate, network, env; trunc = truncrank(k))

        @test prod(gated) ≈ U * S * Vt rtol = eps(real(T))^(1 / 3)
    end

    @testset "apply_operators applies a sequence" begin
        rng = StableRNG(123)
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        g1 = randn_operator(rng, T, (site_axes[2], site_axes[3]))
        g2 = randn_operator(rng, T, (site_axes[3], site_axes[4]))
        gated, _ = apply_operators([g1, g2], network, env)
        @test prod(gated) ≈ apply(g2, apply(g1, prod(network))) rtol =
            eps(real(T))^(1 / 3)
    end

    @testset "message roots gauge a vertex and undo it" begin
        rng = StableRNG(123)
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))
        rtol = eps(real(T))^(1 / 3)

        for (edge, v) in ((2 => 3, 3), (3 => 2, 2))
            sqrt_message, invsqrt_message = sqrth_invsqrth_safe(env[edge])
            @test apply(invsqrt_message, apply(sqrt_message, network[v])) ≈ network[v] rtol =
                rtol
        end
    end
end
