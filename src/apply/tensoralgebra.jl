# Local stand-ins for a general regularized pseudo-inverse, split across
# the two upstream namespaces it's intended to live in:
#
#   * `MAK.inv_regularized(A::AbstractMatrix, tol; kwargs...)`
#     already exists upstream as the matrix-layer pseudo-inverse.
#
#   * `inv_regularized(A::AbstractArray, ::Val; kwargs...)` (N-d unnamed) is
#     defined here in this package's namespace. Intended to move into
#     `TensorAlgebra.jl` as `TA.inv_regularized`, alongside its
#     existing `TA.svd` / `TA.qr` overload set.
#
#   * `MAK.inv_regularized(a::AbstractNamedDimsArray, ...)` is
#     added here, extending MAK's function directly for named arrays.
#     Intended to move into `NamedDimsArrays.jl` (mirroring how NDA already
#     extends `TA.svd` for named arrays).
#
# Until those PRs land, this file is the in-place stand-in. Splitting the
# named overload onto `MAK.inv_regularized` keeps the named and unnamed
# layers in distinct function namespaces (avoiding cross-layer dispatch
# ambiguity) and matches the planned upstream landing.

import MatrixAlgebraKit as MAK
import TensorAlgebra as TA
using LinearAlgebra: I
using NamedDimsArrays: AbstractNamedDimsArray, denamed, dimnames, name, nameddims, randname

# === N-d / TensorAlgebra layer ===

function inv_regularized(
        style::TA.FusionStyle, A::AbstractArray, ndims_codomain::Val;
        tol = nothing, kwargs...
    )
    A_mat = TA.matricize(style, A, ndims_codomain)
    tol_value = isnothing(tol) ? MAK.defaulttol(A_mat) : tol
    Ainv_mat = MAK.inv_regularized(A_mat, tol_value; kwargs...)
    biperm = TA.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = TA.blocks(axes(A)[biperm])
    return TA.unmatricize(style, Ainv_mat, axes_domain, axes_codomain)
end
function inv_regularized(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return inv_regularized(TA.FusionStyle(A), A, ndims_codomain; kwargs...)
end

# === NamedDimsArrays layer (extends `MAK.inv_regularized`) ===

function MAK.inv_regularized(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain_names = collect(name.(dimnames_codomain))
    domain_names = collect(name.(dimnames_domain))
    biperm = TA.blockedperm_indexin(
        Tuple.((dimnames(a), codomain_names, domain_names))...
    )
    perm_codomain, perm_domain = TA.blocks(biperm)
    A_perm = TA.bipermutedims(denamed(a), perm_codomain, perm_domain)
    Ainv_denamed = inv_regularized(A_perm, Val(length(perm_codomain)); kwargs...)
    return nameddims(Ainv_denamed, [domain_names; codomain_names])
end

# Short form: supply the codomain dimnames; the domain is inferred as the
# complement. Matches the 2-arg convention used by `TA.qr` / `TA.lq` /
# `TA.factorize` / `TA.orth` / `TA.polar` for named arrays
# (see `NamedDimsArrays/src/tensoralgebra.jl`).
function MAK.inv_regularized(
        a::AbstractNamedDimsArray, dimnames_codomain; kwargs...
    )
    codomain_names = name.(dimnames_codomain)
    domain_names = setdiff(dimnames(a), codomain_names)
    return MAK.inv_regularized(a, codomain_names, domain_names; kwargs...)
end

# === identity_map ===
#
# 2k-leg identity *map* (pairwise δ per (co_i, dom_i)):
# `I_{co_1, dom_1} ⊗ … ⊗ I_{co_k, dom_k}` reshaped to a 2k-leg tensor.
#
# Local stand-in: dense-only. Eventual home is `TensorAlgebra.jl` with
# an `AbstractNamedDimsArray` overload and axis-type dispatch for the
# graded / FusionTensor specializations (see
# `gate_application/Overview.md` in `ITensorDevelopmentPlans`).

function identity_map(::Type{T}, codomain_axes, domain_axes) where {T}
    co_axes = Tuple(codomain_axes)
    dom_axes = Tuple(domain_axes)
    co_lens = length.(co_axes)
    dom_lens = length.(dom_axes)
    n_co = prod(co_lens; init = 1)
    n_dom = prod(dom_lens; init = 1)
    return reshape(Matrix{T}(I, n_co, n_dom), (co_lens..., dom_lens...))
end

# Note: the BP simple-update `√S` split uses NDA's existing
# `Base.sqrt(::AbstractNamedDimsArray, codomain_dimnames,
# domain_dimnames)` (matrix sqrt as a single named array) directly,
# combined with explicit `replacedimnames` at the call site to split
# the result into two factors sharing a fresh bond. See the comment in
# `apply_gate_bp_nsite!` (Val{2} method) for the call-site
# choreography. A tuple-returning `factorize_sqrt` primitive — splitting
# a Hermitian PSD `M` into `(X, Y)` with a fresh shared bond — was
# previously staged here as a local stand-in but isn't needed for the
# current `√S` use case (K=1 codomain). It can be reintroduced when a
# multi-codomain (K>1) factorization use case lands, alongside the
# rest of the `factorize_<backend>` family
# (`factorize_balanced_eigh`, `factorize_cholesky`) discussed in
# `gate_application/Overview.md` in `ITensorDevelopmentPlans`.
