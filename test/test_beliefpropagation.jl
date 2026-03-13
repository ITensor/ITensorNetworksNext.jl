using DiagonalArrays: δ
using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, src, vertices
using ITensorBase: ITensor, Index, noprime, prime
using ITensorNetworksNext:
    ITensorNetworksNext, BeliefPropagationCache, TensorNetwork, scalar
using LinearAlgebra: LinearAlgebra
using NamedDimsArrays: inds, name
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges, vertextype
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using Test: @test, @testset

function spin_ice_tensornetwork(g)
    links = Dictionary(
        edges(g),
        [Index(2) for e in edges(g)]
        # [Index(2; tags = "edge " => "e$(src(e))_$(dst(e))") for e in edges(g)]
    )
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))

    ts = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        t_data = zeros(Int, 2, 2, 2, 2)
        for (i, j, k, l) in Iterators.product(0:1, 0:1, 0:1, 0:1)
            if i + j + k + l == 2
                t_data[i + 1, j + 1, k + 1, l + 1] = 1
            end
        end
        linkinds = [links[e] for e in es]
        t = t_data[linkinds...]
        set!(ts, v, t)
    end
    return TensorNetwork(g, ts)
end

@testset "BeliefPropagation" begin

    #Chain of tensors
    dims = (2, 1)
    g = named_grid(dims)
    l = Dict(e => Index(2) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = TensorNetwork(g) do v
        is = map(e -> l[e], incident_edges(g, v))
        return randn(Tuple(is))
    end

    bpc = BeliefPropagationCache(tn)
    bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 1)
    z_bp = scalar(bpc)
    z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
    @test z_bp ≈ z_exact atol = 1.0e-14

    #Tree of tensors
    dims = (4, 3)
    g = named_comb_tree(dims)
    l = Dict(e => Index(3) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = TensorNetwork(g) do v
        is = map(e -> l[e], incident_edges(g, v))
        return randn(Tuple(is))
    end

    bpc = BeliefPropagationCache(tn)
    bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 1)
    z_bp = scalar(bpc)
    z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
    @test z_bp ≈ z_exact atol = 1.0e-10

    #Spin Ice Model
    for n in (3, 4, 5)
        dims = (n, n)
        g = named_grid(dims; periodic = true)
        tn = spin_ice_tensornetwork(g)

        bpc = ITensorNetworksNext.BeliefPropagationCache(tn)
        bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 1)

        z_bp = scalar(bpc)

        @test z_bp ≈ 1.5^(n^2)
    end
end
