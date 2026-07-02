module ITensorNetworksNextOMEinsumContractionOrdersExt

using ITensorNetworksNext: ITensorNetworksNext
using OMEinsumContractionOrders: CodeOptimizer

function ITensorNetworksNext.contraction_order(alg::CodeOptimizer, tn)
    return ITensorNetworksNext._contraction_order(alg, tn)
end

end
