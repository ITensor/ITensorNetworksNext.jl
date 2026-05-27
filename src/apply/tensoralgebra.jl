import MatrixAlgebraKit as MAK
import TensorAlgebra as TA
using LinearAlgebra: Diagonal, I, diag
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsOperator, codomainnames,
    denamed, dimnames, domainnames, name, nameddims, operator, randname, setname, state

pinv_tol(λ, pinv::NamedTuple) = pinv_tol(λ; pinv...)
function pinv_tol(
        λ; atol = zero(eltype(λ)),
        rtol = iszero(atol) ? eps(eltype(λ)) * length(λ) : zero(eltype(λ))
    )
    return max(atol, rtol * maximum(abs, λ; init = zero(eltype(λ))))
end

sqrt_safe(a::Number, tol = MAK.defaulttol(a)) = abs(a) < tol ? zero(a) : sqrt(a)

# Gram factorization of a PSD matrix `M ≈ X' * X` via its eigendecomposition,
# laid out like the factorizations in `TensorAlgebra` / `NamedDimsArrays`:
# self-contained matrix primitives, an `AbstractArray` layer that
# matricizes/permutes (`FusionStyle`/`Val`, integer-permutation, and label
# entries), and a named layer that delegates to the label entry and re-wraps
# the results. `gram_eigh_full` returns the forward factor `X = Diagonal(sqrtλ)
# * V'` (rank leg first); `gram_eigh_full_with_pinv` additionally returns
# `Y ≈ pinv(X)` (rank leg last), so that `X * Y ≈ I`. They are separate
# codepaths (different factor counts / leg layouts); the dispatch forwarders and
# operator entry, identical for both, are `@eval`-generated.

function gram_eigh_full(A::AbstractMatrix; alg = nothing, pinv = (;))
    D, V = MAK.eigh_full(A, MAK.select_algorithm(MAK.eigh_full, A, alg))
    λ = diag(D)
    sqrtλ = map(l -> sqrt_safe(l, pinv_tol(λ, pinv)), λ)
    return Diagonal(sqrtλ) * V'
end
function gram_eigh_full_with_pinv(A::AbstractMatrix; alg = nothing, pinv = (;))
    D, V = MAK.eigh_full(A, MAK.select_algorithm(MAK.eigh_full, A, alg))
    λ = diag(D)
    sqrtλ = map(l -> sqrt_safe(l, pinv_tol(λ, pinv)), λ)
    inv_sqrtλ = map(s -> iszero(s) ? s : inv(s), sqrtλ)
    return Diagonal(sqrtλ) * V', V * Diagonal(inv_sqrtλ)
end

function gram_eigh_full(
        style::TA.FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    Xmat = gram_eigh_full(TA.matricize(style, A, ndims_codomain); kwargs...)
    biperm = TA.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain = first(TA.blocks(axes(A)[biperm]))
    return TA.unmatricize(style, Xmat, (axes(Xmat, 1),), axes_codomain)
end
function gram_eigh_full_with_pinv(
        style::TA.FusionStyle, A::AbstractArray, ndims_codomain::Val; kwargs...
    )
    Xmat, Ymat = gram_eigh_full_with_pinv(TA.matricize(style, A, ndims_codomain); kwargs...)
    biperm = TA.trivialbiperm(ndims_codomain, Val(ndims(A)))
    axes_codomain = first(TA.blocks(axes(A)[biperm]))
    rank_axis = axes(Xmat, 1)
    return TA.unmatricize(style, Xmat, (rank_axis,), axes_codomain),
        TA.unmatricize(style, Ymat, axes_codomain, (rank_axis,))
end

function gram_eigh_full(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    X = gram_eigh_full(denamed(a), dimnames(a), codomain, domain; kwargs...)
    rank_name = randname(dimnames(a, 1))
    return nameddims(X, (rank_name, codomain...))
end
function gram_eigh_full_with_pinv(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    X, Y = gram_eigh_full_with_pinv(denamed(a), dimnames(a), codomain, domain; kwargs...)
    rank_name = randname(dimnames(a, 1))
    return nameddims(X, (rank_name, codomain...)), nameddims(Y, (codomain..., rank_name))
end

# `FusionStyle` convenience, label entry, and operator entry are identical for
# both factorizations. (No standalone integer-permutation method: it would be
# ambiguous with the named-array method, since named arrays subtype
# `AbstractArray`; the label entry permutes inline instead.)
for f in (:gram_eigh_full, :gram_eigh_full_with_pinv)
    @eval begin
        function $f(A::AbstractArray, ndims_codomain::Val; kwargs...)
            return $f(TA.FusionStyle(A), A, ndims_codomain; kwargs...)
        end
        function $f(A::AbstractArray, labels_A, labels_codomain, labels_domain; kwargs...)
            biperm = TA.blockedperm_indexin(
                Tuple.((labels_A, labels_codomain, labels_domain))...
            )
            perm_codomain, perm_domain = TA.blocks(biperm)
            A_perm = TA.bipermutedims(A, perm_codomain, perm_domain)
            return $f(A_perm, Val(length(perm_codomain)); kwargs...)
        end
        function $f(M::AbstractNamedDimsOperator; kwargs...)
            return $f(state(M), codomainnames(M), domainnames(M); kwargs...)
        end
    end
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
