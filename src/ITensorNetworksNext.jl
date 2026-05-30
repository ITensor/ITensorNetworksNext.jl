module ITensorNetworksNext

# Imported as a name only so that the `[compat] TensorAlgebra = "0.9.2"` floor
# (needed for a `bipermutedimsopadd!` fix in `TensorAlgebra` 0.9.2 that affects
# `NamedDimsArrays`-mediated tensor multiplication) isn't reported as a stale
# dependency by Aqua.
using TensorAlgebra: TensorAlgebra

include("select_algorithm.jl")
include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyNamedDimsArrays/LazyNamedDimsArrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")
include("operator_init.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/messagecache_constructors.jl")
include("beliefpropagation/beliefpropagation.jl")

include("apply/apply_operators.jl")

end
