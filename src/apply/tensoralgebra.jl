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
using NamedDimsArrays: AbstractNamedDimsArray, denamed, dimnames, name, nameddims
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
    domain_names = Tuple(setdiff(dimnames(a), codomain_names))
    return MAK.inv_regularized(a, codomain_names, domain_names; kwargs...)
end
