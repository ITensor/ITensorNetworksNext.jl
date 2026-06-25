using Graphs: add_edge!, edges, vertices
using ITensorBase: Index, name, nameddims, setname, uniquename
using ITensorNetworksNext: ITensorNetwork, TensorNetworkOperator, dmrg, insertlink!
using LinearAlgebra: eigen
using MatrixAlgebraKit: truncrank
using NamedGraphs.NamedGraphGenerators: named_path_graph
using NamedGraphs: NamedGraph
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset

# Transverse-field Ising model `H = -J Σ Z Z - h Σ X`, built by hand as operator networks
# and as dense matrices for exact-diagonalization references.

const X = [0.0 1.0; 1.0 0.0]
const Z = [1.0 0.0; 0.0 -1.0]
const Id = [1.0 0.0; 0.0 1.0]

kron_ops(ops) = foldl(kron, ops)

# Bulk Ising MPO tensor `W[a, b, ket, bra]` (bond dim 3 finite-state machine).
function ising_mpo_tensor(; J, h)
    W = zeros(3, 3, 2, 2)
    W[1, 1, :, :] = Id
    W[2, 1, :, :] = Z
    W[3, 1, :, :] = -h * X
    W[3, 2, :, :] = -J * Z
    W[3, 3, :, :] = Id
    return W
end

# Path-graph TFIM as a `TensorNetworkOperator` on vertices `1:N`.
function tfim_path_operator(N, sites, sitemap; J, h)
    verts = collect(1:N)
    bond_edges = [(verts[i], verts[i + 1]) for i in 1:(N - 1)]
    bonds = Dict(
        e => setname(Index(Base.OneTo(3)), uniquename(name(sites[verts[1]])))
            for e in bond_edges
    )
    W = ising_mpo_tensor(; J, h)
    g = NamedGraph(verts)
    for (v1, v2) in bond_edges
        add_edge!(g, v1, v2)
    end
    function tensor(v)
        k = name(sites[v])
        b = sitemap[k]
        left = findfirst(e -> e[2] == v, bond_edges)
        right = findfirst(e -> e[1] == v, bond_edges)
        if isnothing(left)            # left boundary: row 3 of W
            data = W[3, :, :, :]
            return nameddims(data, (name(bonds[bond_edges[right]]), k, b))
        elseif isnothing(right)       # right boundary: column 1 of W
            data = W[:, 1, :, :]
            return nameddims(data, (name(bonds[bond_edges[left]]), k, b))
        end
        return nameddims(
            W, (name(bonds[bond_edges[left]]), name(bonds[bond_edges[right]]), k, b)
        )
    end
    tn = ITensorNetwork(g) do v
        return tensor(v)
    end
    return TensorNetworkOperator(tn, sitemap)
end

function tfim_path_dense(N; J, h)
    H = zeros(2^N, 2^N)
    for i in 1:(N - 1)
        H += -J * kron_ops([(j == i || j == i + 1) ? Z : Id for j in 1:N])
    end
    for i in 1:N
        H += -h * kron_ops([j == i ? X : Id for j in 1:N])
    end
    return H
end

# Star tree: center vertex 1 bonded to leaves 2, 3, 4.
function tfim_star_operator(sites, sitemap; J, h)
    leaves = [2, 3, 4]
    g = NamedGraph(collect(1:4))
    for v in leaves
        add_edge!(g, 1, v)
    end
    bonds = Dict(
        v => setname(Index(Base.OneTo(3)), uniquename(name(sites[1]))) for v in leaves
    )
    # Center tensor: one bond per leaf, with `-hX` once and `-JZ`/`I` on each leaf channel.
    center = zeros(3, 3, 3, 2, 2)
    center[1, 1, 1, :, :] = -h * X
    for (i, _) in enumerate(leaves)
        center[ntuple(j -> j == i ? 2 : 1, 3)..., :, :] = -J * Z
        center[ntuple(j -> j == i ? 3 : 1, 3)..., :, :] = Id
    end
    center_names = (
        name(bonds[2]), name(bonds[3]), name(bonds[4]), name(sites[1]),
        sitemap[name(sites[1])],
    )
    function leaf_tensor(v)
        data = zeros(3, 2, 2)
        data[1, :, :] = Id
        data[2, :, :] = Z
        data[3, :, :] = -h * X
        return nameddims(data, (name(bonds[v]), name(sites[v]), sitemap[name(sites[v])]))
    end
    tensors =
        Dict(1 => nameddims(center, center_names), (v => leaf_tensor(v) for v in leaves)...)
    tn = ITensorNetwork(g) do v
        return tensors[v]
    end
    return TensorNetworkOperator(tn, sitemap)
end

function tfim_star_dense(; J, h)
    H = zeros(16, 16)
    for v in [2, 3, 4]
        H += -J * kron_ops([(j == 1 || j == v) ? Z : Id for j in 1:4])
    end
    for i in 1:4
        H += -h * kron_ops([j == i ? X : Id for j in 1:4])
    end
    return H
end

function random_ket(rng, g)
    sites = Dict(v => Index(Base.OneTo(2)) for v in vertices(g))
    ket = ITensorNetwork(NamedGraph(collect(vertices(g)))) do v
        return randn(rng, Float64, (sites[v],))
    end
    for e in edges(g)
        insertlink!(ket, e)
    end
    return ket, sites
end

@testset "dmrg" begin
    @testset "path TFIM matches ED (N=$N)" for N in (4, 6)
        rng = StableRNG(8)
        g = named_path_graph(N)
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_path_operator(N, sites, sitemap; J = 1.0, h = 0.7)
        exact = minimum(eigen(tfim_path_dense(N; J = 1.0, h = 0.7)).values)

        _, energy = dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 8))
        @test energy ≈ exact rtol = 1.0e-8
    end

    @testset "energy-tolerance early stop" begin
        N = 6
        rng = StableRNG(8)
        g = named_path_graph(N)
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_path_operator(N, sites, sitemap; J = 1.0, h = 0.7)
        exact = minimum(eigen(tfim_path_dense(N; J = 1.0, h = 0.7)).values)

        _, energy =
            dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 50, tol = 1.0e-10))
        @test energy ≈ exact rtol = 1.0e-8
    end

    @testset "per-sweep truncation schedule" begin
        N = 6
        rng = StableRNG(8)
        g = named_path_graph(N)
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_path_operator(N, sites, sitemap; J = 1.0, h = 0.7)
        exact = minimum(eigen(tfim_path_dense(N; J = 1.0, h = 0.7)).values)

        # Ramp the bond dimension up over sweeps; the final rank is exact for N=6.
        trunc = n -> truncrank(min(2^n, 8))
        _, energy = dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 8), trunc)
        @test energy ≈ exact rtol = 1.0e-8
    end

    @testset "energy decreases with more sweeps" begin
        N = 6
        rng = StableRNG(8)
        g = named_path_graph(N)
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_path_operator(N, sites, sitemap; J = 1.0, h = 0.7)
        exact = minimum(eigen(tfim_path_dense(N; J = 1.0, h = 0.7)).values)

        _, energy1 = dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 1))
        _, energy4 = dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 4))
        @test energy4 ≤ energy1 + 1.0e-10
        @test energy4 ≥ exact - 1.0e-10
    end

    @testset "star-tree TFIM matches ED" begin
        rng = StableRNG(11)
        g = NamedGraph(collect(1:4))
        for v in 2:4
            add_edge!(g, 1, v)
        end
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_star_operator(sites, sitemap; J = 1.0, h = 0.6)
        exact = minimum(eigen(tfim_star_dense(; J = 1.0, h = 0.6)).values)

        _, energy = dmrg(operator, ket0; stopping_criterion = (; maxsweeps = 8))
        @test energy ≈ exact rtol = 1.0e-8
    end

    @testset "stopping_criterion is required" begin
        rng = StableRNG(8)
        g = named_path_graph(4)
        ket0, sites = random_ket(rng, g)
        sitemap = Dict(name(sites[v]) => uniquename(name(sites[v])) for v in vertices(g))
        operator = tfim_path_operator(4, sites, sitemap; J = 1.0, h = 0.7)
        @test_throws ArgumentError dmrg(operator, ket0)
    end
end
