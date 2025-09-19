using Graphs: AbstractGraph, AbstractEdge, dst, edges, rem_edge!, src
using NamedGraphs.GraphsExtensions: incident_edges, rem_edges!

# Remove all edges incident to vertex `v` in graph `g`.
function rem_incident_edges!(g::AbstractGraph, v)
  rem_edges!(g, incident_edges(g, v))
  return g
end

# TODO: Move to `NamedGraphs.GraphsExtensions`,
# replace `DataGraphs.arrange`.
# TODO: Only arrange if the graph is undirected.
function is_edge_arranged(e::AbstractEdge)
  return src(e) < dst(e)
end
function is_edge_arranged(g::AbstractGraph, e::AbstractEdge)
  return is_edge_arranged(e)
end
function arrange_edge(e::AbstractEdge)
  return is_edge_arranged(e) ? e : reverse(e)
end
function arrange_edge(g::AbstractGraph, e::AbstractEdge)
  return is_edge_arranged(g, e) ? e : reverse(e)
end
function arranged_edges(g::AbstractGraph)
  return map(e -> arrange_edge(g, e), edges(g))
end
