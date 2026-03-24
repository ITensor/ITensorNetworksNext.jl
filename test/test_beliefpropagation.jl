using DiagonalArrays: δ
using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, src, vertices
using ITensorBase: ITensor, Index, noprime, prime
using ITensorNetworksNext:
    ITensorNetworksNext, BeliefPropagationCache, TensorNetwork, partitionfunction
using LinearAlgebra: LinearAlgebra
using NamedDimsArrays: inds, name
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges, vertextype
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using Test: @test, @testset

function ising_tensornetwork(g::AbstractGraph, β::Real; h = 0.0)
    links = Dictionary(
        edges(g),
        [Index(2; tags = "edge" => "e$(src(e))_$(dst(e))") for e in edges(g)]
    )
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))

    # symmetric sqrt of Boltzmann matrix W = exp(β σσ')
    sqrt_Ws = Dictionary()
    for e in edges(g)
        W = [exp(-(β + 2 * h)) exp(β); exp(β) exp(-(β - 2 * h))]

        F = LinearAlgebra.svd(W)
        U, S, V = F.U, F.S, F.Vt
        @assert U * LinearAlgebra.diagm(S) * V ≈ W
        id = [1.0 0.0; 0.0 1.0]
        set!(sqrt_Ws, e, id)
        set!(sqrt_Ws, reverse(e), U * LinearAlgebra.diagm(S) * V)
    end
    ts = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        #t = ITensor(1.0, physical_inds[v]...) * delta([links[e] for e in es])
        t = δ(Float64, Tuple([links[e] for e in es]))
        for e in es
            t_prime = ITensor(sqrt_Ws[e], (name(links[e]), name(prime(links[e])))) * t
            newinds = noprime.(inds(t_prime))
            t = ITensor(parent(t_prime), name.(newinds))
        end
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
    z_bp = partitionfunction(bpc)
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
    z_bp = partitionfunction(bpc)
    z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
    @test z_bp ≈ z_exact atol = 1.0e-10

    #Square lattice Ising model
    dims = (3, 3)
    g = named_grid(dims)
    tn = ising_tensornetwork(g, 0.05, h = 0.5)
    bpc = ITensorNetworksNext.BeliefPropagationCache(tn)
    bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 50, tol = 1.0e-10)

    z_bp = partitionfunction(bpc)
    z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
    @test z_bp ≈ z_exact rtol = 1.0e-4
end
