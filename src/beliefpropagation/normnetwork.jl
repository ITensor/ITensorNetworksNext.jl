using Graphs: dst, edges, edgetype, src
using ITensorBase: codomainnames, denamed, domainnames, name, operator, replacedimnames,
    similar_operator, state, uniquename
using NamedGraphs.GraphsExtensions: all_edges, incident_edges
using SplitApplyCombine: mapmany

function message_environment(::UndefInitializer, nn::NormNetwork)
    messages = mapmany(vertices(nn)) do vertex
        return map(in_incident_edges(nn, vertex)) do edge
            braview = BraView(nn)
            ketview = KetView(nn)

            ketnames = linknames(KetView(nn), edge)

            brainds = linkinds(braview, edge)
            branames = name.(brainds)
            braaxis = denamed.(brainds)

            # Message axis is conj to the tensor it points to.
            message = similar_operator(ketview[vertex], braaxis, branames, ketnames)

            return edge => message
        end
    end

    return messagecache(messages)
end

function message_environment(f::Base.Callable, nn::NormNetwork)
    return map(f, message_environment(undef, nn))
end

function beliefpropagation(nn::NormNetwork, messages; kwargs...)
    renamed_messages = map(messages) do msg
        if !any(name -> has_indname(KetView(nn), name), dimnames(msg))
            error(
                "provided message on does not have have any index \
                names in common with the tensor network contained in the norm."
            )
        end

        bramap = Dict(codomainnames(msg) .=> Base.Fix1(namemap, nn).(domainnames(msg)))

        return replacedimnames(name -> get(bramap, name, name), state(msg))
    end

    cache = _beliefpropagation(nn, renamed_messages; kwargs...)

    # Re-wrap each converged message as an operator with codomain = bra names and
    # domain = ket names from the map.
    return messagecache(keys(cache)) do edge
        ketnames = linknames(KetView(nn), edge)
        branames = linknames(BraView(nn), edge)
        return operator(cache[edge], branames, ketnames)
    end
end
