# Local stand-ins for a general regularized pseudo-inverse, layered like
# `TensorAlgebra`'s binary factorizations (`svd`, `qr`, …):
#
#   * `AbstractMatrix` — thin adapter over `MatrixAlgebraKit.inv_regularized`
#     that exposes its positional `tol` as a kwarg, so the layers above can
#     forward kwargs uniformly.
#
#   * `AbstractArray` (`Val{ndims_codomain}` / perm / labelled) — interprets
#     `A` with axes `(codomain..., domain...)` as a linear map
#     `domain → codomain` and returns the pseudo-inverse map
#     `codomain → domain`, i.e. an array with axes `(domain..., codomain...)`.
#
#   * `AbstractNamedDimsArray` — same shape, resolved through dim names
#     (matching the `TensorAlgebra.svd` named overload's API in NamedDimsArrays).
#
# Intended to move upstream into `TensorAlgebra.jl` and `NamedDimsArrays.jl`
# (one PR each) before this branch merges; this file is the in-place
# stand-in until those land.

using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, denamed, dimnames, name, nameddims
using TensorAlgebra: TensorAlgebra

# === Matrix layer ===

function inv_regularized(
        A::AbstractMatrix; tol = MatrixAlgebraKit.defaulttol(A), kwargs...
    )
    return MatrixAlgebraKit.inv_regularized(A, tol; kwargs...)
end

# === N-d / TensorAlgebra layer ===

function inv_regularized(
        style::TensorAlgebra.FusionStyle, A::AbstractArray, ndims_codomain::Val;
        kwargs...
    )
    A_mat = TensorAlgebra.matricize(style, A, ndims_codomain)
    Ainv_mat = inv_regularized(A_mat; kwargs...)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain, axes_domain = TensorAlgebra.blocks(axes(A)[biperm])
    axes_Ainv = TensorAlgebra.tuplemortar((axes_domain, axes_codomain))
    return TensorAlgebra.unmatricize(style, Ainv_mat, axes_Ainv)
end
function inv_regularized(A::AbstractArray, ndims_codomain::Val; kwargs...)
    return inv_regularized(TensorAlgebra.FusionStyle(A), A, ndims_codomain; kwargs...)
end

function inv_regularized(
        A::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
        kwargs...
    )
    A_perm = TensorAlgebra.bipermutedims(A, perm_codomain, perm_domain)
    return inv_regularized(A_perm, Val(length(perm_codomain)); kwargs...)
end

function inv_regularized(
        A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...
    )
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((labels_A, labels_codomain, labels_domain))...
    )
    return inv_regularized(A, TensorAlgebra.blocks(biperm)...; kwargs...)
end

# === NamedDimsArrays layer ===

function inv_regularized(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain_names = name.(dimnames_codomain)
    domain_names = name.(dimnames_domain)
    ainv_denamed = inv_regularized(
        denamed(a), dimnames(a), codomain_names, domain_names; kwargs...
    )
    return nameddims(ainv_denamed, (domain_names..., codomain_names...))
end
