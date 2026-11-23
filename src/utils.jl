using Dictionaries: AbstractIndices, Dictionary

# `map` does something clever to figure out the element type of the output when not
# inferable, but the `map` overload on `Dictionary` does not, so we fix this here.
narrow_map(f, v) = map(f, v)
narrow_map(f, v::AbstractIndices) = Dictionary(v, [f(x) for x in v])
