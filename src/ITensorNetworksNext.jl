module ITensorNetworksNext

# Imported as a name only so the floor `[compat] TensorAlgebra = "0.9.5"`
# (needed for the `gram_eigh_full` convention flip and the new
# `TensorAlgebra.one` worker) isn't reported as a stale dependency by Aqua.
using TensorAlgebra: TensorAlgebra

include("select_algorithm.jl")
include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyNamedDimsArrays/LazyNamedDimsArrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagation.jl")
include("beliefpropagation/normnetwork.jl")

include("apply/apply_operators.jl")

end
