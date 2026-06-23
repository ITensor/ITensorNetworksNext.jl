module ITensorNetworksNext

include("select_algorithm.jl")
include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("normnetwork.jl")
include("normnetworkview.jl")
include("ITensorNetworkGenerators/ITensorNetworkGenerators.jl")
include("contract_network.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagation.jl")
include("beliefpropagation/normnetwork.jl")

include("apply/apply_operators.jl")

end
