module ITensorNetworksNext

include("LazyNamedDimsArrays/LazyNamedDimsArrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")
include("abstract_problem.jl")
include("iterators.jl")
include("adapters.jl")

end
