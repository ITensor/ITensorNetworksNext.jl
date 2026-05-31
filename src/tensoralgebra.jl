using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, dimnames, domainnames, name, nameddims, operator, randname, setname, state
using Random: Random
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, FusionStyle, bipermutedims,
    blockedperm_indexin, blocks, matricize, trivialbiperm, unmatricize

# Local stand-ins for what would eventually become upstream interface functions in
# `TensorAlgebra` / `NamedDimsArrays`. Naming:
#
#   - `similar_operator(prototype, [T,] codomain)` — eventual
#     `TensorAlgebra.similar_operator` / `NamedDimsArrays.similar_operator`.
#   - `one_tensor(a, ...)` — eventual `TensorAlgebra.one` (paralleling `TA.svd`,
#     `TA.eigen`).
#   - `one_operator(op)` / `one_operator(na, codomain, domain)` — eventual methods
#     of `Base.one` on `AbstractNamedDimsOperator` and `AbstractNamedDimsArray`.
#     Held under the local name `one_operator` until then to avoid piracy on
#     `NamedDimsArrays` types.
#   - `randn_operator!([rng,] op)` / `rand_operator!([rng,] op)` — eventual methods
#     of `Random.randn!` / `Random.rand!` on `AbstractNamedDimsOperator`. Held locally
#     for the same piracy reason, plus to hide the workaround for the ITensor
#     `eltype(::Type) === Any` issue (peeling to the concrete storage so the
#     stdlib `randn!` / `rand!` sees the runtime eltype).
#   - `dag`, `dual` — no-op stubs for the tensor and axis involutions.

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

# === Identity operator/tensor: TA-style layered API ===
#
# Mirrors `TensorAlgebra.svd` / `eigen`: a chain of dispatches accepting named
# operators, named arrays with codomain/domain names, raw arrays with labels, with
# biperms, with perms, or in canonical `(codomain..., domain...)` layout — all funnel
# into the canonical worker `one_tensor(style, a, ndims_codomain::Val)`, which
# matricizes the array, calls `MatrixAlgebraKit.one!` on the matrix, and unmatricizes
# back.
#
# All forms are out-of-place: `a` is treated as a shape prototype, not mutated. We
# rely on `matricize` returning a fresh non-aliasing array; a future view-returning
# `matricized` would be the lower-level building block for an in-place variant.

# --- Named layers (local `one_operator`; would be `Base.one` upstream) ---

function one_operator(op::AbstractNamedDimsOperator)
    co, dom = codomainnames(op), domainnames(op)
    return operator(one_operator(state(op), co, dom), co, dom)
end

function one_operator(na::AbstractNamedDimsArray, codomain_names, domain_names)
    raw = one_tensor(denamed(na), dimnames(na), codomain_names, domain_names)
    return nameddims(raw, dimnames(na))
end

# --- Raw-array layers (`one_tensor`; would be `TensorAlgebra.one` upstream) ---

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

# Canonical form: matricize → matrix-level identity → unmatricize.
function one_tensor(a::AbstractArray, ndims_codomain::Val)
    return one_tensor(FusionStyle(a), a, ndims_codomain)
end
function one_tensor(style::FusionStyle, a::AbstractArray, ndims_codomain::Val)
    a_mat = matricize(style, a, ndims_codomain)
    MatrixAlgebraKit.one!(a_mat)
    biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
    axes_codomain, axes_domain = blocks(axes(a)[biperm])
    return unmatricize(style, a_mat, axes_codomain, axes_domain)
end

# === Random fills for operators ===
#
# Local helpers that would eventually become methods of `Random.randn!` and
# `Random.rand!` on `AbstractNamedDimsOperator`. They hide the workaround for the
# ITensor `eltype(typeof(::ITensor)) === Any` issue: a direct `randn!(op)` / `rand!(op)`
# dispatches on `Type{Any}` and fails, so we peel down to the concrete storage where
# the runtime eltype is honored.

function randn_operator!(op::AbstractNamedDimsOperator)
    return randn_operator!(Random.default_rng(), op)
end
function randn_operator!(rng::Random.AbstractRNG, op::AbstractNamedDimsOperator)
    Random.randn!(rng, denamed(state(op)))
    return op
end

function rand_operator!(op::AbstractNamedDimsOperator)
    return rand_operator!(Random.default_rng(), op)
end
function rand_operator!(rng::Random.AbstractRNG, op::AbstractNamedDimsOperator)
    Random.rand!(rng, denamed(state(op)))
    return op
end
