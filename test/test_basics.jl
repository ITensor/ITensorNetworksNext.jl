using Dictionaries: Indices
using Graphs: edges, vertices
using ITensorNetworksNext: IndsNetwork, TensorNetwork
using ITensorBase: Index
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

@testset "ITensorNetworksNext" begin
  dims = (3, 3)
  g = named_grid(dims)
  vs = map(_ -> Index(2), Indices(vertices(g)))
  es = map(_ -> Index(1), Indices(edges(g)))
  is = IndsNetwork(g, vs, es)
  x = TensorNetwork(is)
end
