using Dictionaries: Dictionary
using ITensorBase: Index
using ITensorNetworksNext: ITensorNetworksNext, BeliefPropagationCache, TensorNetwork, adapt_messages, default_message, default_messages, edge_scalars, messages, setmessages!, factors, freenergy,
  partitionfunction
using Graphs: edges, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using Test: @test, @testset

@testset "BeliefPropagation" begin
    dims = (4, 1)
    g = named_grid(dims)
    l = Dict(e => Index(2) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = TensorNetwork(g) do v
      is = map(e -> l[e], incident_edges(g, v))
      return randn(Tuple(is))
    end

    bpc = BeliefPropagationCache(tn)
    bpc = ITensorNetworksNext.update(bpc; maxiter = 10)
    z_bp = partitionfunction(bpc)
    z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
    @test abs(z_bp - z_exact) <= 1e-14
end