using Dictionaries: Indices
using Graphs: dst, edges, has_edge, ne, nv, src, vertices
using ITensorNetworksNext: TensorNetwork, linkinds, siteinds
using ITensorBase: Index
using NamedDimsArrays: dimnames
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

@testset "ITensorNetworksNext" begin
  @testset "Construct TensorNetwork product state" begin
    dims = (3, 3)
    g = named_grid(dims)
    s = Dict(v => Index(2) for v in vertices(g))
    tn = TensorNetwork(g) do v
      return randn(s[v])
    end
    @test nv(tn) == 9
    @test ne(tn) == ne(g)
    @test issetequal(vertices(tn), vertices(g))
    @test issetequal(arranged_edges(tn), arranged_edges(g))
    for v in vertices(tn)
      @test siteinds(tn, v) == [s[v]]
    end
    for v1 in vertices(tn)
      for v2 in vertices(tn)
        v1 == v2 && continue
        haslink = !isempty(linkinds(tn, v1 => v2))
        @test haslink == has_edge(tn, v1 => v2)
      end
    end
    for e in edges(tn)
      @test isone(length(only(linkinds(tn, e))))
    end
  end
  @testset "Construct TensorNetwork partition function" begin
    dims = (3, 3)
    g = named_grid(dims)
    l = Dict(e => Index(2) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = TensorNetwork(g) do v
      is = map(e -> l[e], incident_edges(g, v))
      return randn(Tuple(is))
    end
    @test nv(tn) == 9
    @test ne(tn) == ne(g)
    @test issetequal(vertices(tn), vertices(g))
    @test issetequal(arranged_edges(tn), arranged_edges(g))
    for v in vertices(tn)
      @test isempty(siteinds(tn, v))
    end
    for v1 in vertices(tn)
      for v2 in vertices(tn)
        v1 == v2 && continue
        haslink = !isempty(linkinds(tn, v1 => v2))
        @test haslink == has_edge(tn, v1 => v2)
      end
    end
    for e in edges(tn)
      @test only(linkinds(tn, e)) == l[e]
    end
  end
end
