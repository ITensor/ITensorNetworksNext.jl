module ITensorNetworksNext

# Imported as a name only so that the `[compat] TensorAlgebra = "0.9.2"` floor
# (needed for a `bipermutedimsopadd!` fix in `TensorAlgebra` 0.9.2 that affects
# `NamedDimsArrays`-mediated tensor multiplication) isn't reported as a stale
# dependency by Aqua.
using TensorAlgebra: TensorAlgebra

include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyNamedDimsArrays/LazyNamedDimsArrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagation.jl")

include("ITensorNetworksNextParallel/ITensorNetworksNextParallel.jl")

end
