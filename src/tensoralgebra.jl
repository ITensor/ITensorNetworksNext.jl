using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, domainnames, name, operator, randname, setname, state

# Local stand-ins for upstream `TensorAlgebra.similar_operator` /
# `NamedDimsArrays.similar_operator` / `id_operator` /
# `Base.one(::AbstractNamedDimsOperator)`, intended to move into
# `TensorAlgebra` / `NamedDimsArrays`.

# Tensor-algebra interface no-op stubs. Currently identity; backends (graded sectors,
# complex tensors, etc.) will overload these for their semantics.
#
# `dag` is the involution on TENSORS (conjugate-transpose, sector-direction flip, …).
# `dual` is the involution on AXES (vector space → dual vector space).
dag(x) = x
dual(x) = x

# Allocate a square operator with the given `codomain` named axes. Domain axes are
# derived as `dual.(codomain)` with fresh `randname`-generated names; backend / device
# inherited from `prototype` via `Base.similar`.
function similar_operator(prototype, ::Type{T}, codomain) where {T}
    domain_names = randname.(name.(codomain))
    domain_axes = setname.(dual.(codomain), domain_names)
    raw = similar(prototype, T, (codomain..., domain_axes...))
    return operator(raw, name.(codomain), domain_names)
end
function similar_operator(prototype, codomain)
    return similar_operator(prototype, eltype(prototype), codomain)
end

# === Identity operator: layered flow ===
#
#   Operator              (Base.one)
#     → NamedDimsArray    (id_operator)
#       → AbstractArray   (via `_matricize`, currently a `reshape` view)
#         → Matrix        (MatrixAlgebraKit.one!)
#
# The matrix-level `one!` mutates a `reshape` view of the underlying storage, so the
# data propagates back up the layers automatically.

# Operator layer: allocate a new operator with the same codomain/domain structure as
# `op`, filled with the identity map. Codomain and domain names are preserved.
function Base.one(op::AbstractNamedDimsOperator)
    return id_operator(state(op), codomainnames(op), domainnames(op))
end

# NamedDimsArray layer: `prototype` is shaped like `(codomain..., domain...)`. Allocate
# a fresh same-shape named array, fill it with the matricized identity, and wrap as an
# operator with the given codomain/domain names.
function id_operator(prototype::AbstractNamedDimsArray, codomain_names, domain_names)
    a = similar(prototype)
    MatrixAlgebraKit.one!(_matricize(denamed(a), length(codomain_names)))
    return operator(a, codomain_names, domain_names)
end

# AbstractArray layer: view `a` as a matrix with its first `K` axes flattened to rows
# and the remaining axes flattened to columns. Dense-only — graded backends need a
# sector-aware matricize.
function _matricize(a::AbstractArray, K::Int)
    co_dim = prod(ntuple(i -> size(a, i), K))
    dom_dim = prod(ntuple(i -> size(a, K + i), ndims(a) - K))
    return reshape(a, co_dim, dom_dim)
end
