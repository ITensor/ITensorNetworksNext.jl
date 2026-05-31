using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, dimnames, domainnames, name, nameddims, operator, randname, setname, state
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, FusionStyle, bipermutedims,
    blockedperm_indexin, blocks, matricize, trivialbiperm, unmatricize

# Local stand-ins for upstream `TensorAlgebra.similar_operator` /
# `NamedDimsArrays.similar_operator` / `TensorAlgebra.one` /
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

# === Identity tensor: TA-style layered API ===
#
# Mirrors `TensorAlgebra.svd` / `eigen`: a chain of dispatches accepting (named arrays
# with names, raw arrays with labels, with biperms, with perms, or in canonical
# (codomain..., domain...) layout) all funnel into the in-place canonical worker
# `one_tensor!(a, ndims_codomain::Val)`, which matricizes the array, calls
# `MatrixAlgebraKit.one!`, and unmatricizes back.
#
# `one_tensor` is the local name for what would eventually be `TensorAlgebra.one`.
#
# Named layers extend `Base.one` (piracy on `AbstractNamedDimsArray` /
# `AbstractNamedDimsOperator`); raw-array layers live in `one_tensor` /
# `one_tensor!`.

# --- Named layers ---

function Base.one(op::AbstractNamedDimsOperator)
    co, dom = codomainnames(op), domainnames(op)
    return operator(one(state(op), co, dom), co, dom)
end

function Base.one(na::AbstractNamedDimsArray, codomain_names, domain_names)
    raw = one_tensor(denamed(na), dimnames(na), codomain_names, domain_names)
    return nameddims(raw, dimnames(na))
end

# --- Raw-array layers ---

# Label form: derive a biperm from per-axis labels.
function one_tensor(a::AbstractArray, labels_a, labels_codomain, labels_domain)
    biperm = blockedperm_indexin(Tuple.((labels_a, labels_codomain, labels_domain))...)
    return one_tensor(a, blocks(biperm)...)
end

# Biperm form.
function one_tensor(a::AbstractArray, biperm::AbstractBlockPermutation{2})
    return one_tensor(a, blocks(biperm)...)
end

# Explicit codomain/domain permutation form: physically permute axes into canonical
# layout, then dispatch to the canonical form.
function one_tensor(
        a::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}},
        perm_domain::Tuple{Vararg{Int}}
    )
    a_perm = bipermutedims(a, perm_codomain, perm_domain)
    return one_tensor(a_perm, Val(length(perm_codomain)))
end

# Canonical form (out-of-place): allocate a fresh similar buffer and fill.
function one_tensor(a::AbstractArray, ndims_codomain::Val)
    return one_tensor!(similar(a), ndims_codomain)
end

# Canonical-form worker (in-place): matricize → matrix-level identity → unmatricize.
function one_tensor!(a::AbstractArray, ndims_codomain::Val)
    return one_tensor!(FusionStyle(a), a, ndims_codomain)
end
function one_tensor!(style::FusionStyle, a::AbstractArray, ndims_codomain::Val)
    a_mat = matricize(style, a, ndims_codomain)
    MatrixAlgebraKit.one!(a_mat)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
    axes_codomain, axes_domain = blocks(axes(a)[biperm])
    return unmatricize(style, a_mat, axes_codomain, axes_domain)
end
