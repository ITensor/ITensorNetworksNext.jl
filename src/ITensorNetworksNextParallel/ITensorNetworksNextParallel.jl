module ITensorNetworksNextParallel

using ..ITensorNetworksNext: BeliefPropagationCache
using Graphs: add_vertex!, neighbors, vertices
using NamedGraphs.GraphsExtensions: subgraph
using NamedGraphs.PartitionedGraphs: QuotientVertex

"""
    get_subiterate(subproblem::AI.Problem, subalgorithm::AI.Algorithm, state::AI.State)

For a given `subproblem` and `subalgorithm` of a parent nested algorithm,
derive (from the parent state `state`) the iterate to be used in the associated sub state.
The returned value of this function is then pass to a remote call of `initialize_state`.
"""
get_subiterate(::AI.Problem, ::AI.Algorithm, state::AI.State) = state.iterate

include("dagger.jl")

end # ITensorNetworksNextParallel
