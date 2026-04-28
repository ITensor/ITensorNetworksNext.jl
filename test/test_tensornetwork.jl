using DataGraphs: assigned_edge_data, assigned_vertex_data
using Graphs: dst, edges, has_edge, ne, nv, src, vertices
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: similar_graph
using Test: @test, @testset

@testset "`TensorNetwork`" begin
    @testset "DataGraphs/NamedGraphs interface" begin
        dims = (3, 3)
        g = named_grid(dims)
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
            return randn(s[v])
        end

        stn = similar_graph(tn)
        @test stn isa TensorNetwork
        @test vertices(stn) == vertices(tn)
        @test edges(stn) == edges(tn)
        @test isempty(assigned_vertex_data(stn))
        @test isempty(assigned_edge_data(stn))

        stn = similar_graph(tn, vertices(tn))
        @test vertices(stn) == vertices(tn)
        @test ne(stn) == 0
        @test isempty(assigned_vertex_data(stn))
        @test isempty(assigned_edge_data(stn))
    end
end
