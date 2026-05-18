using LinearAlgebra: Hermitian, adjoint, diag, diagm, eigen
using MatrixAlgebraKit: MatrixAlgebraKit
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, nameddims, randname
using TensorAlgebra: TensorAlgebra

"""
    invert_diagonal_message(env::AbstractNamedDimsArray, codomain, domain; tol=0)

Inverse of a 2-leg diagonal `env` with names `(codomain..., domain...)`, returned
as a 2-leg named array with names `(domain..., codomain...)` (flipped, so it can
be contracted to undo a gauge-in). Regularized via `MatrixAlgebraKit.inv_regularized`.
Assumes `env` is diagonal — appropriate for sqrt-message Vidal-gauge caches.
"""
function invert_diagonal_message(env::AbstractNamedDimsArray, codomain, domain; tol = 0)
    codomain_names = name.(codomain)
    domain_names = name.(domain)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(env), codomain_names, domain_names))...
    )
    perm_co, perm_dom = TensorAlgebra.blocks(biperm)
    env_perm = TensorAlgebra.bipermutedims(env.denamed, perm_co, perm_dom)
    σ = diag(env_perm)
    inv_σ = MatrixAlgebraKit.inv_regularized.(σ, tol)
    return nameddims(diagm(inv_σ), (domain_names..., codomain_names...))
end

function balanced_eigh_and_inv(
        A::AbstractMatrix;
        trunc = nothing, tol = 0, ishermitian = true
    )
    F = ishermitian ? eigen(Hermitian(Matrix(A))) : eigen(Matrix(A))
    λ, U = F.values, F.vectors
    if !isnothing(trunc)
        kept = MatrixAlgebraKit.findtruncated(λ, trunc)
        λ = λ[kept]
        U = U[:, kept]
    end
    R = real(eltype(λ))
    sqrtλ = sqrt.(max.(real.(λ), zero(R)))
    invsqrtλ = MatrixAlgebraKit.inv_regularized.(sqrtλ, tol)
    Uᴴ = adjoint(U)
    Y = sqrtλ .* Uᴴ
    Yinv = U .* transpose(invsqrtλ)
    return Y, Yinv
end

function balanced_eigh_and_inv(A::AbstractArray, ndims_codomain::Val; kwargs...)
    style = TensorAlgebra.FusionStyle(A)
    A_mat = TensorAlgebra.matricize(style, A, ndims_codomain)
    Y_mat, Yinv_mat = balanced_eigh_and_inv(A_mat; kwargs...)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    _, axes_dom = TensorAlgebra.blocks(axes(A)[biperm])
    ax_bond = (axes(Y_mat, 1),)
    axes_Y = TensorAlgebra.tuplemortar((ax_bond, axes_dom))
    axes_Yinv = TensorAlgebra.tuplemortar((axes_dom, ax_bond))
    Y = TensorAlgebra.unmatricize(style, Y_mat, axes_Y)
    Yinv = TensorAlgebra.unmatricize(style, Yinv_mat, axes_Yinv)
    return Y, Yinv
end

function balanced_eigh_and_inv(
        A::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
        kwargs...
    )
    A_perm = TensorAlgebra.bipermutedims(A, perm_codomain, perm_domain)
    return balanced_eigh_and_inv(A_perm, Val(length(perm_codomain)); kwargs...)
end

function balanced_eigh_and_inv(P::AbstractNamedDimsArray, codomain, domain; kwargs...)
    codomain_names = name.(codomain)
    domain_names = name.(domain)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(P), codomain_names, domain_names))...
    )
    perm_co, perm_dom = TensorAlgebra.blocks(biperm)
    Y_d, Yinv_d = balanced_eigh_and_inv(P.denamed, perm_co, perm_dom; kwargs...)
    bond_name = randname(first(domain_names))
    Y = nameddims(Y_d, (bond_name, domain_names...))
    Yinv = nameddims(Yinv_d, (domain_names..., bond_name))
    return Y, Yinv
end

"""
    svd_compact_named(A; trunc=nothing)
    svd_compact_named(A, ndims_codomain::Val; trunc=nothing)
    svd_compact_named(A, perm_codomain, perm_domain; trunc=nothing)
    svd_compact_named(A, codomain, domain; trunc=nothing)

Like `MatrixAlgebraKit.svd_compact` / `svd_trunc`, but for `(Abstract)NamedDimsArray`
inputs returns `(U, σ, V)` where `U` has names `(codomain..., bond_name)`,
`V` has names `(bond_name, domain...)`, and `σ` is the singular-value
`Vector`. A single `bond_name` is shared by `U` and `V` (unlike
`TensorAlgebra.svd`, which inserts a 2-leg singular-value matrix with two
distinct bond names).
"""
function svd_compact_named(A::AbstractMatrix; trunc = nothing)
    U, S, Vᴴ = if isnothing(trunc)
        MatrixAlgebraKit.svd_compact(Matrix(A))
    else
        MatrixAlgebraKit.svd_trunc(Matrix(A); trunc)
    end
    return U, diag(S), Vᴴ
end

function svd_compact_named(A::AbstractArray, ndims_codomain::Val; kwargs...)
    style = TensorAlgebra.FusionStyle(A)
    A_mat = TensorAlgebra.matricize(style, A, ndims_codomain)
    U_mat, σ, V_mat = svd_compact_named(A_mat; kwargs...)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_co, axes_dom = TensorAlgebra.blocks(axes(A)[biperm])
    ax_bond = (axes(U_mat, 2),)
    axes_U = TensorAlgebra.tuplemortar((axes_co, ax_bond))
    axes_V = TensorAlgebra.tuplemortar((ax_bond, axes_dom))
    U = TensorAlgebra.unmatricize(style, U_mat, axes_U)
    V = TensorAlgebra.unmatricize(style, V_mat, axes_V)
    return U, σ, V
end

function svd_compact_named(
        A::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
        kwargs...
    )
    A_perm = TensorAlgebra.bipermutedims(A, perm_codomain, perm_domain)
    return svd_compact_named(A_perm, Val(length(perm_codomain)); kwargs...)
end

function svd_compact_named(A::AbstractNamedDimsArray, codomain, domain; kwargs...)
    codomain_names = name.(codomain)
    domain_names = name.(domain)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(A), codomain_names, domain_names))...
    )
    perm_co, perm_dom = TensorAlgebra.blocks(biperm)
    U_d, σ, V_d = svd_compact_named(A.denamed, perm_co, perm_dom; kwargs...)
    bond_name = randname(first(codomain_names))
    U = nameddims(U_d, (codomain_names..., bond_name))
    V = nameddims(V_d, (bond_name, domain_names...))
    return U, σ, V
end

function balanced_svd(A::AbstractMatrix; trunc = nothing)
    U, S, Vᴴ = if isnothing(trunc)
        MatrixAlgebraKit.svd_compact(Matrix(A))
    else
        MatrixAlgebraKit.svd_trunc(Matrix(A); trunc)
    end
    σ = diag(S)
    sqrtσ = sqrt.(σ)
    X = U .* transpose(sqrtσ)
    Y = sqrtσ .* Vᴴ
    return X, Y
end

function balanced_svd(A::AbstractArray, ndims_codomain::Val; kwargs...)
    style = TensorAlgebra.FusionStyle(A)
    A_mat = TensorAlgebra.matricize(style, A, ndims_codomain)
    X_mat, Y_mat = balanced_svd(A_mat; kwargs...)
    biperm = TensorAlgebra.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_co, axes_dom = TensorAlgebra.blocks(axes(A)[biperm])
    ax_bond = (axes(X_mat, 2),)
    axes_X = TensorAlgebra.tuplemortar((axes_co, ax_bond))
    axes_Y = TensorAlgebra.tuplemortar((ax_bond, axes_dom))
    X = TensorAlgebra.unmatricize(style, X_mat, axes_X)
    Y = TensorAlgebra.unmatricize(style, Y_mat, axes_Y)
    return X, Y
end

function balanced_svd(
        A::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}};
        kwargs...
    )
    A_perm = TensorAlgebra.bipermutedims(A, perm_codomain, perm_domain)
    return balanced_svd(A_perm, Val(length(perm_codomain)); kwargs...)
end

function balanced_svd(A::AbstractNamedDimsArray, codomain, domain; kwargs...)
    codomain_names = name.(codomain)
    domain_names = name.(domain)
    biperm = TensorAlgebra.blockedperm_indexin(
        Tuple.((dimnames(A), codomain_names, domain_names))...
    )
    perm_co, perm_dom = TensorAlgebra.blocks(biperm)
    X_d, Y_d = balanced_svd(A.denamed, perm_co, perm_dom; kwargs...)
    bond_name = randname(first(codomain_names))
    X = nameddims(X_d, (codomain_names..., bond_name))
    Y = nameddims(Y_d, (bond_name, domain_names...))
    return X, Y
end
