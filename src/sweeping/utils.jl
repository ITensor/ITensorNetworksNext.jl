# Utility functions for processing keyword arguments.
function repeat_last(v::AbstractVector, len::Int)
    return [v; fill(v[end], max(len - length(v), 0))]
end
repeat_last(v, len::Int) = fill(v, len)
function extend_columns(nt::NamedTuple, len::Int)
    return (; (keys(nt) .=> map(v -> repeat_last(v, len), values(nt)))...)
end
rowlength(nt::NamedTuple) = only(unique(length.(values(nt))))
function rows(nt::NamedTuple, len::Int = rowlength(nt))
    return [(; (keys(nt) .=> map(v -> v[i], values(nt)))...) for i in 1:len]
end
