module ITensorNetworksNext

if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public apply_operator, apply_operators"
        )
    )
end

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

include("apply/apply_operators.jl")

end
