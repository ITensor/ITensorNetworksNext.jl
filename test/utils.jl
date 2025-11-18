module TestUtils
using QuadGK: quadgk
# Exact critical inverse temperature for 2D square lattice Ising model.
βc_2d_ising(elt::Type{<:Number} = Float64) = elt(log(1 + √2) / 2)
# Exact infinite volume free energy density for 2D square lattice Ising model.
function f_2d_ising(β::Real; J::Real = one(β))
    κ = 2sinh(2β * J) / cosh(2β * J)^2
    integrand(θ) = log((1 + √(abs(1 - (κ * sin(θ))^2))) / 2)
    integral, _ = quadgk(integrand, 0, π)
    return (-log(2cosh(2β * J)) - (1 / (2π)) * integral) / β
end
function f_1d_ising(β::Real; J::Real = one(β), h::Real = zero(β))
    λ⁺ = exp(β * J) * (cosh(β * h) + √(sinh(β * h)^2 + exp(-4β * J)))
    return -(log(λ⁺) / β)
end
function f_1d_ising(β::Real, N::Integer; periodic::Bool = true, kwargs...)
    return if periodic
        f_1d_ising_periodic(β, N; kwargs...)
    else
        f_1d_ising_open(β, N; kwargs...)
    end
end
function f_1d_ising_periodic(β::Real, N::Integer; J::Real = one(β), h::Real = zero(β))
    r = √(sinh(β * h)^2 + exp(-4β * J))
    λ⁺ = exp(β * J) * (cosh(β * h) + r)
    λ⁻ = exp(β * J) * (cosh(β * h) - r)
    Z = λ⁺^N + λ⁻^N
    return -(log(Z) / (β * N))
end
function f_1d_ising_open(β::Real, N::Integer; J::Real = one(β), h::Real = zero(β))
    isone(N) && return 2cosh(β * h)
    T = [
        exp(β * (J + h)) exp(-β * J);
        exp(-β * J) exp(β * (J - h));
    ]
    b = [exp(β * h / 2), exp(-β * h / 2)]
    Z = (b' * (T^(N - 1)) * b)[]
    return -(log(Z) / (β * N))
end
end
