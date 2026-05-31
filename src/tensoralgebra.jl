using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, dimnames, domainnames, inds, name, nameddims, operator, randname, setname,
    state

# Local stand-ins for upstream `TensorAlgebra.similar_operator` /
# `NamedDimsArrays.similar_operator` / `Base.one(::AbstractNamedDimsOperator)` /
# `LinearAlgebra.one!(::AbstractNamedDimsOperator)`, intended to move into
# `TensorAlgebra` / `NamedDimsArrays`.

# Allocate a square operator with the given `codomain` named axes. Domain axes are
# derived as `dag.(codomain)` with fresh `randname`-generated names; backend / device
# inherited from `prototype` via `Base.similar`.
function similar_operator(prototype, ::Type{T}, codomain) where {T}
    domain_names = randname.(name.(codomain))
    domain_axes = setname.(dag.(codomain), domain_names)
    raw = similar(prototype, T, (codomain..., domain_axes...))
    return operator(raw, name.(codomain), domain_names)
end
function similar_operator(prototype, codomain)
    return similar_operator(prototype, eltype(prototype), codomain)
end

# In-place identity fill. Reshape the underlying data to a (codomain × domain) matrix
# and call `MAK.one!`. Returns `a`.
#
# Dense-only for now: for a `GradedArray`-backed operator the reshape is not the right
# matricization, so this would produce a non-sector-aware identity. The upstream version
# will route through `TA.matricize` / `MAK.diagview` to handle graded backings correctly.
function MatrixAlgebraKit.one!(a::AbstractNamedDimsOperator)
    raw = denamed(state(a))
    K = length(codomainnames(a))
    co_dims = ntuple(i -> size(raw, i), K)
    dom_dims = ntuple(i -> size(raw, K + i), ndims(raw) - K)
    M = reshape(raw, prod(co_dims), prod(dom_dims))
    MatrixAlgebraKit.one!(M)
    return a
end

# Allocate-and-fill identity from a prototype operator. Same codomain (and matching
# auto-named domain) as `a`, eltype taken from `a`.
function Base.one(a::AbstractNamedDimsOperator)
    raw_inds = collect(inds(state(a)))
    K = length(codomainnames(a))
    codomain_axes = ntuple(i -> raw_inds[i], K)
    return MatrixAlgebraKit.one!(similar_operator(state(a), eltype(a), codomain_axes))
end
