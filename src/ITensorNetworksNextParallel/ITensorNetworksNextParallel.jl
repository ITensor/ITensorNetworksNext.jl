module ITensorNetworksNextParallel

using Graphs: neighbors, add_vertex!, vertices
using NamedGraphs.GraphsExtensions: subgraph
using NamedGraphs.PartitionedGraphs: QuotientVertex
using ..ITensorNetworksNext: BeliefPropagationCache

subcache(cache::BeliefPropagationCache, vertex::QuotientVertex) = subcache(cache, vertices(cache, vertex))
function subcache(cache::BeliefPropagationCache, vertices)
    subcache = subgraph(cache, vertices)

    for vertex in vertices
        for neighbor_vertex in neighbors(cache, vertex)
            add_vertex!(subcache, neighbor_vertex)
            # Add in necessary messages.
            subcache[vertex => neighbor_vertex] = cache[vertex => neighbor_vertex]
            subcache[neighbor_vertex => vertex] = cache[neighbor_vertex => vertex]
        end
    end

    return subcache
end

include("distributed.jl")
include("dagger.jl")

end # ITensorNetworksNextParallel
