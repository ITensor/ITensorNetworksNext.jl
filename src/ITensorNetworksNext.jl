module ITensorNetworksNext

include("select_algorithm.jl")
include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyITensors/LazyITensors.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("normnetwork.jl")
include("normnetworkview.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagation.jl")
include("beliefpropagation/normnetwork.jl")

include("apply/apply_operators.jl")

end
