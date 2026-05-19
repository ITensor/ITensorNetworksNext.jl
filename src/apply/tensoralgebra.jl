# Local stand-ins for a general regularized pseudo-inverse, split across
# the two upstream namespaces it's intended to live in:
#
#   * `MAK.inv_regularized(A::AbstractMatrix, tol; kwargs...)`
#     already exists upstream as the matrix-layer pseudo-inverse.
#
#   * `inv_regularized(A::AbstractArray, ::Val; kwargs...)` (N-d unnamed) is
#     defined here in this package's namespace. Intended to move into
#     `TensorAlgebra.jl` as `TensorAlgebra.inv_regularized`, alongside its
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
using LinearAlgebra: I
using NamedDimsArrays: AbstractNamedDimsArray, denamed, dimnames, name, nameddims, randname
using TensorAlgebra: TensorAlgebra

# === N-d / TensorAlgebra layer ===

function inv_regularized(
        style::TensorAlgebra.FusionStyle, A::AbstractArray, ndims_codomain::Val;
        tol = nothing, kwargs...
    )
    A_mat = TensorAlgebra.matricize(style, A, ndims_codomain)
    tol_value = isnothing(tol) ? MAK.defaulttol(A_mat) : tol
    Ainv_mat = MAK.inv_regularized(A_mat, tol_value; kwargs...)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = TensorAlgebra.blocks(axes(A)[biperm])
    axes_Ainv = TensorAlgebra.tuplemortar((axes_domain, axes_codomain))
    return TensorAlgebra.unmatricize(style, Ainv_mat, axes_Ainv)
end
function inv_regularized(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return inv_regularized(TensorAlgebra.FusionStyle(A), A, ndims_codomain; kwargs...)
end

# === NamedDimsArrays layer (extends `MAK.inv_regularized`) ===

function MAK.inv_regularized(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain_names = name.(dimnames_codomain)
    domain_names = name.(dimnames_domain)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(a), codomain_names, domain_names))...
    )
    perm_codomain, perm_domain = TensorAlgebra.blocks(biperm)
    A_perm = TensorAlgebra.bipermutedims(denamed(a), perm_codomain, perm_domain)
    Ainv_denamed = inv_regularized(A_perm, Val(length(perm_codomain)); kwargs...)
    return nameddims(Ainv_denamed, (domain_names..., codomain_names...))
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

# === sqrt_factorization ===
#
# Factor a PSD named array `a` as `(X, Y)` with `X * Y ≈ a` via named
# contraction, where `X` and `Y` share a fresh-named bond. For
# k-codomain input, `X` has names `(codomain..., new_bond)` and `Y`
# has names `(new_bond, domain...)`.
#
# Layered through `TA.matricize` → matrix `sqrt` → `TA.unmatricize`,
# matching the shape of `inv_regularized` above. The N-d / TA layer
# is namespaced locally (intended TensorAlgebra.sqrt_factorization),
# the named layer extends here.

function sqrt_factorization(
        style::TensorAlgebra.FusionStyle, A::AbstractArray, ndims_codomain::Val
    )
    M = TensorAlgebra.matricize(style, A, ndims_codomain)
    sqrtM = sqrt(M)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = TensorAlgebra.blocks(axes(A)[biperm])
    bond_axis = axes(sqrtM, 2)
    axes_X = TensorAlgebra.tuplemortar((axes_codomain, (bond_axis,)))
    axes_Y = TensorAlgebra.tuplemortar(((bond_axis,), axes_domain))
    return (
        TensorAlgebra.unmatricize(style, sqrtM, axes_X),
        TensorAlgebra.unmatricize(style, sqrtM, axes_Y),
    )
end

function sqrt_factorization(
        a::AbstractNamedDimsArray, codomain_dimnames, domain_dimnames
    )
    codomain_names = name.(codomain_dimnames)
    domain_names = name.(domain_dimnames)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(a), codomain_names, domain_names))...
    )
    perm_codomain, perm_domain = TensorAlgebra.blocks(biperm)
    A_perm = TensorAlgebra.bipermutedims(denamed(a), perm_codomain, perm_domain)
    style = TensorAlgebra.FusionStyle(A_perm)
    X_denamed, Y_denamed = sqrt_factorization(style, A_perm, Val(length(perm_codomain)))
    new_bond = randname(first(codomain_names))
    return (
        nameddims(X_denamed, (codomain_names..., new_bond)),
        nameddims(Y_denamed, (new_bond, domain_names...)),
    )
end

function sqrt_factorization(a::AbstractNamedDimsArray, codomain_dimnames)
    codomain_names = name.(codomain_dimnames)
    domain_names = setdiff(dimnames(a), codomain_names)
    return sqrt_factorization(a, codomain_names, domain_names)
end
