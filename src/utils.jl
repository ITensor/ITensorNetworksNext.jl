using Dictionaries: AbstractIndices, Dictionary
using ITensorBase: ITensorBase, IndexName, settags
using UUIDs: uuid5

# `map` does something clever to figure out the element type of the output when not
# inferable, but the `map` overload on `Dictionary` does not, so we fix this here.
narrow_map(f, v) = map(f, v)
narrow_map(f, v::AbstractIndices) = Dictionary(v, [f(x) for x in v])

"""
    derivedname(name::IndexName, key) -> IndexName

Return a name derived deterministically from `name` and `key`: the identifier is replaced by
the name-based UUID `uuid5(uuid(name), string(key))`, `key` is added as a bare-label tag, and
the other tags and the prime level are kept. Every process derives the same name from the same
`name` and `key`, and the result differs from `name` and from any independently minted name.
"""
function derivedname(name::IndexName, key)
    derived = ITensorBase.setuuid(name, uuid5(ITensorBase.uuid(name), string(key)))
    return settags(derived, string(key))
end
