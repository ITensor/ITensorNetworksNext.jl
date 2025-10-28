module ITensorNetworksNext

include("lazynameddimsarrays.jl")
include("abstracttensornetwork.jl")
include("tensornetwork.jl")
include("contract_network.jl")
include("abstract_problem.jl")
include("iterators.jl")

include("beliefpropagation/abstractbeliefpropagationcache.jl")
include("beliefpropagation/beliefpropagationcache.jl")
include("beliefpropagation/beliefpropagationproblem.jl")

end
