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
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, dimnames, domainnames, name, nameddims, operator, randname, setname, state

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

function similar_operator(prototype::AbstractNamedDimsArray, codomain_axes)
    co_axes = Tuple(codomain_axes)
    dom_axes = setname.(co_axes, randname.(name.(co_axes)))
    A = similar(denamed(prototype), (co_axes..., dom_axes...))
    return operator(A, collect(name.(co_axes)), collect(name.(dom_axes)))
end

function Base.one(a::AbstractNamedDimsOperator)
    co = codomainnames(a)
    dom = domainnames(a)
    A = state(a)
    A_denamed = denamed(A)
    style = TA.FusionStyle(A_denamed)
    ndims_co = Val(length(co))
    A_mat = TA.matricize(style, A_denamed, ndims_co)
    id_mat = similar(A_mat)
    copyto!(id_mat, I)
    biperm = TA.trivialbiperm(ndims_co, Val(ndims(A_denamed)))
    co_axes, dom_axes = TA.blocks(axes(A_denamed)[biperm])
    id_denamed = TA.unmatricize(style, id_mat, co_axes, dom_axes)
    id_nda = nameddims(id_denamed, dimnames(A))
    return operator(id_nda, co, dom)
end
