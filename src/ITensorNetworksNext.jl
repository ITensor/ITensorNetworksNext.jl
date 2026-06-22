module ITensorNetworksNext

if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public apply_operator, apply_operators, beliefpropagation_normnetwork, identity_norm_message_env, normnetwork, norm_message_env, ones_norm_message_env, rand_norm_message_env, randn_norm_message_env, similar_norm_message_env"
        )
    )
end

include("select_algorithm.jl")
include("AlgorithmsInterfaceExtensions/AlgorithmsInterfaceExtensions.jl")
include("LazyITensors/LazyITensors.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("normnetwork.jl")
include("TensorNetworkGenerators/TensorNetworkGenerators.jl")
include("contract_network.jl")

include("beliefpropagation/messagecache.jl")
include("beliefpropagation/beliefpropagation.jl")
include("beliefpropagation/normnetwork.jl")

include("apply/apply_operators.jl")

end
