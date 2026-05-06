module ITensorNetworksNext

include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyNamedDimsArrays/LazyNamedDimsArrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")
include("sweeping/utils.jl")
include("sweeping/eigenproblem.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagationproblem.jl")

end
